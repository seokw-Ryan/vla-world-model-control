"""Shared OpenVLA inference wrapper for use with Isaac Sim and Isaac Lab."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class OpenVLAConfig:
    """Configuration for the OpenVLA model."""

    model_path: str = "openvla/openvla-7b"
    unnorm_key: str = "bridge_orig"
    prompt_template: str = "In: {instruction}\nOut:"
    image_size: int = 224
    dtype: str = "bfloat16"
    load_in_4bit: bool = True
    # Action scaling
    position_scale: float = 1.0
    rotation_scale: float = 1.0
    # Gripper thresholds
    gripper_threshold: float = 0.5
    gripper_open_value: float = 1.0
    gripper_close_value: float = -1.0


@dataclass
class VLAAction:
    """Structured action output from the VLA model."""

    delta_pos: np.ndarray  # (3,) xyz delta position
    delta_rot: np.ndarray  # (3,) euler angle delta rotation
    gripper: float  # binary: 1.0 (open) or 0.0 (closed)
    gripper_continuous: float  # raw continuous gripper value
    raw: np.ndarray  # (7,) raw action vector from model


class OpenVLAWrapper:
    """Reusable wrapper around OpenVLA for robot control.

    Separates __init__ (config-only) from load() (GPU/model init) because
    Isaac Sim requires SimulationApp to be created before any torch/CUDA calls.
    """

    def __init__(self, config: Optional[OpenVLAConfig] = None):
        self.config = config or OpenVLAConfig()
        self._model = None
        self._processor = None
        self._device = None
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> "OpenVLAWrapper":
        """Load the model and processor onto the GPU. Call after SimulationApp init."""
        import torch
        from transformers.generation import GenerationMixin
        from transformers import AutoConfig, AutoProcessor, BitsAndBytesConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        import transformers.tokenization_utils as tokenization_utils
        import transformers.tokenization_utils_base as tokenization_utils_base

        # OpenVLA's remote processor code expects these symbols under
        # transformers.tokenization_utils, but Isaac Sim's bundled transformers
        # exposes them from tokenization_utils_base instead.
        for name in ("PaddingStrategy", "PreTokenizedInput", "TextInput", "TruncationStrategy"):
            if not hasattr(tokenization_utils, name) and hasattr(tokenization_utils_base, name):
                setattr(tokenization_utils, name, getattr(tokenization_utils_base, name))

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

        print(f"[OpenVLA] Loading {self.config.model_path} on {self._device} "
              f"(dtype={self.config.dtype}, 4bit={self.config.load_in_4bit})...")

        self._processor = AutoProcessor.from_pretrained(
            self.config.model_path, trust_remote_code=True
        )

        config = AutoConfig.from_pretrained(self.config.model_path, trust_remote_code=True)
        if hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"

        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "config": config,
        }
        if self._device == "cuda" and self.config.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )
            model_kwargs["device_map"] = "auto"

        class_ref = None
        if hasattr(config, "auto_map") and isinstance(config.auto_map, dict):
            class_ref = config.auto_map.get("AutoModelForVision2Seq")
        if not class_ref:
            raise ValueError(
                "OpenVLA config is missing auto_map['AutoModelForVision2Seq']; "
                "cannot resolve the remote model class in this transformers runtime."
            )

        model_class = get_class_from_dynamic_module(
            class_ref,
            self.config.model_path,
            trust_remote_code=True,
        )
        # Isaac Sim bundles a newer transformers runtime than OpenVLA expects.
        # Force eager attention and provide a safe class attribute so the parent
        # PreTrainedModel init path does not touch the remote property's
        # `self.language_model` before that field exists.
        model_class._supports_sdpa = False
        if not issubclass(model_class, GenerationMixin):
            model_class = type(
                model_class.__name__,
                (model_class, GenerationMixin),
                {},
            )
        if "recompute_mapping" not in getattr(model_class.tie_weights, "__code__", ()).co_varnames:
            original_tie_weights = model_class.tie_weights

            def compat_tie_weights(self, *args, **kwargs):
                return original_tie_weights(self)

            model_class.tie_weights = compat_tie_weights
        self._model = model_class.from_pretrained(
            self.config.model_path, **model_kwargs
        )
        if not getattr(self._model, "hf_device_map", None):
            self._model = self._model.to(self._device)
        self._model.eval()

        self._loaded = True
        print("[OpenVLA] Model loaded successfully.")
        return self

    def predict(self, image: np.ndarray, instruction: str) -> VLAAction:
        """Run inference on a single image + instruction pair.

        Args:
            image: RGB or RGBA numpy array (any size, will be preprocessed).
            instruction: Natural language task instruction.

        Returns:
            VLAAction with structured delta action.
        """
        import torch

        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        pil_image = self._preprocess_image(image)
        prompt = self.config.prompt_template.format(instruction=instruction)

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

        inputs = self._processor(prompt, pil_image).to(self._device, dtype=torch_dtype)

        with torch.inference_mode():
            raw_action = self._model.predict_action(
                **inputs, unnorm_key=self.config.unnorm_key, do_sample=False
            )

        return self._postprocess_action(raw_action)

    def _preprocess_image(self, image: np.ndarray):
        """Convert numpy image to PIL RGB at the expected size."""
        from PIL import Image

        # Handle RGBA (Isaac Sim cameras output RGBA)
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]

        # Ensure uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        pil_image = Image.fromarray(image, "RGB")

        # Resize to expected input size
        if pil_image.size != (self.config.image_size, self.config.image_size):
            pil_image = pil_image.resize(
                (self.config.image_size, self.config.image_size), Image.LANCZOS
            )

        return pil_image

    def _postprocess_action(self, raw_action: np.ndarray) -> VLAAction:
        """Split 7-DoF action into structured components with scaling."""
        raw = np.array(raw_action, dtype=np.float64)

        delta_pos = raw[:3] * self.config.position_scale
        delta_rot = raw[3:6] * self.config.rotation_scale
        gripper_continuous = float(raw[6])

        # Threshold gripper: > threshold → open (1.0), else closed (0.0)
        gripper = 1.0 if gripper_continuous > self.config.gripper_threshold else 0.0

        return VLAAction(
            delta_pos=delta_pos,
            delta_rot=delta_rot,
            gripper=gripper,
            gripper_continuous=gripper_continuous,
            raw=raw,
        )
