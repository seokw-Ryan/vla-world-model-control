#!/usr/bin/env python3
"""Train a LeRobot policy from a public Hugging Face SO100/SO101 dataset.

This is a thin wrapper around ``python -m lerobot.scripts.lerobot_train`` that:

1. inspects a remote LeRobot dataset on the Hugging Face Hub,
2. prints the available camera keys,
3. highlights top-down camera candidates when present, and
4. launches training with SO100/SO101-friendly defaults.

It is intended for datasets collected by other people online, rather than
locally recorded demonstrations.

Examples:
    python scripts/train_hf_so_dataset.py \
        --dataset-repo-id aswinkumar99/LeRobot-SO101-Pick-Place

    python scripts/train_hf_so_dataset.py \
        --dataset-repo-id lerobot/svla_so100_stacking \
        --policy-kind act \
        --batch-size 8 \
        --steps 100000

    python scripts/train_hf_so_dataset.py \
        --dataset-repo-id yangxinye/real_so101_record_v1 \
        --camera-key observation.images.top \
        --rename-camera-to-front \
        --print-only

Notes:
    - This script does not modify the dataset.
    - The existing SO100 sim config in ``configs/sim/so100_standalone.yaml``
      already uses a top-down camera above the XY plane for evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess

from vla_world_model_control.shared import (
    LEROBOT_SRC,
    PROJECT_ROOT,
    add_lerobot_import_paths_to_sys_path,
    extend_pythonpath,
    stderr_log as log,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train from a public Hugging Face SO100/SO101 LeRobot dataset."
    )
    parser.add_argument(
        "--dataset-repo-id",
        required=True,
        help="Public Hugging Face dataset repo id, e.g. user/my_so101_dataset.",
    )
    parser.add_argument(
        "--robot",
        choices=("auto", "so100", "so101"),
        default="auto",
        help="Used only for naming defaults and user-facing logs.",
    )
    parser.add_argument(
        "--policy-kind",
        choices=("xvla", "act"),
        default="xvla",
        help="Training preset to launch. XVLA is the default VLA path.",
    )
    parser.add_argument(
        "--policy-path",
        default=None,
        help="Optional pretrained policy override. Defaults to lerobot/xvla-base for XVLA.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional local root/cache directory for the dataset.",
    )
    parser.add_argument(
        "--dataset-revision",
        default=None,
        help="Optional dataset revision/tag/branch.",
    )
    parser.add_argument(
        "--camera-key",
        default=None,
        help=(
            "Optional dataset camera key to prioritize, e.g. observation.images.top. "
            "Used for validation/logging and optional rename-map generation."
        ),
    )
    parser.add_argument(
        "--rename-camera-to-front",
        action="store_true",
        help=(
            "If --camera-key is set and differs from observation.images.front, add a rename_map "
            "so policies expecting a front camera can reuse the selected top-view camera."
        ),
    )
    parser.add_argument("--device", default="cuda", help="Training device, e.g. cuda or cpu.")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="Policy dtype for XVLA training.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size override.")
    parser.add_argument("--steps", type=int, default=None, help="Training steps override.")
    parser.add_argument("--save-freq", type=int, default=None, help="Checkpoint save frequency override.")
    parser.add_argument("--eval-freq", type=int, default=None, help="Eval frequency override.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--seed", type=int, default=1000, help="Training seed.")
    parser.add_argument("--job-name", default=None, help="Optional LeRobot job name.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory.")
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the resolved lerobot-train command without launching it.",
    )
    return parser.parse_args()
def ensure_lerobot_import_path() -> None:
    add_lerobot_import_paths_to_sys_path()


def is_top_down_camera_key(key: str) -> bool:
    key_lower = key.lower()
    top_tokens = ("top", "overhead", "bird", "ceiling", "xy")
    return any(token in key_lower for token in top_tokens)


def choose_camera_key(camera_keys: list[str], requested_key: str | None) -> tuple[str | None, list[str]]:
    top_candidates = [key for key in camera_keys if is_top_down_camera_key(key)]
    if requested_key:
        return requested_key, top_candidates
    if top_candidates:
        return top_candidates[0], top_candidates
    return (camera_keys[0] if camera_keys else None), top_candidates


def infer_robot_name(dataset_repo_id: str, requested_robot: str) -> str:
    if requested_robot != "auto":
        return requested_robot
    repo_lower = dataset_repo_id.lower()
    if "so100" in repo_lower:
        return "so100"
    if "so101" in repo_lower:
        return "so101"
    return "so101"


def default_job_name(args: argparse.Namespace) -> str:
    robot = infer_robot_name(args.dataset_repo_id, args.robot)
    dataset_slug = args.dataset_repo_id.split("/")[-1].replace(".", "-")
    return f"{args.policy_kind}_{robot}_{dataset_slug}"


def default_output_dir(args: argparse.Namespace, job_name: str) -> str:
    return str(PROJECT_ROOT / "outputs" / "train" / job_name)


def build_train_command(args: argparse.Namespace, rename_map: dict[str, str]) -> list[str]:
    job_name = args.job_name or default_job_name(args)
    output_dir = args.output_dir or default_output_dir(args, job_name)

    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--job_name={job_name}",
        f"--output_dir={output_dir}",
        f"--device={args.device}",
        f"--num_workers={args.num_workers}",
        f"--seed={args.seed}",
    ]

    if args.dataset_root:
        cmd.append(f"--dataset.root={args.dataset_root}")
    if args.dataset_revision:
        cmd.append(f"--dataset.revision={args.dataset_revision}")
    if rename_map:
        cmd.append(f"--rename_map={json.dumps(rename_map)}")

    if args.policy_kind == "xvla":
        cmd.extend(
            [
                f"--policy.path={args.policy_path or 'lerobot/xvla-base'}",
                f"--policy.dtype={args.dtype}",
                "--policy.action_mode=auto",
                "--policy.freeze_vision_encoder=false",
                "--policy.freeze_language_encoder=false",
                "--policy.train_policy_transformer=true",
                "--policy.train_soft_prompts=true",
            ]
        )
        if args.batch_size is not None:
            cmd.append(f"--batch_size={args.batch_size}")
        else:
            cmd.append("--batch_size=2")
        if args.steps is not None:
            cmd.append(f"--steps={args.steps}")
        else:
            cmd.append("--steps=5000")
        if args.save_freq is not None:
            cmd.append(f"--save_freq={args.save_freq}")
        else:
            cmd.append("--save_freq=1000")
        if args.eval_freq is not None:
            cmd.append(f"--eval_freq={args.eval_freq}")
        else:
            cmd.append("--eval_freq=0")
    else:
        cmd.extend(
            [
                "--policy.type=act",
                "--policy.chunk_size=50",
                "--policy.n_action_steps=50",
                "--policy.vision_backbone=resnet18",
            ]
        )
        if args.batch_size is not None:
            cmd.append(f"--batch_size={args.batch_size}")
        else:
            cmd.append("--batch_size=8")
        if args.steps is not None:
            cmd.append(f"--steps={args.steps}")
        else:
            cmd.append("--steps=100000")
        if args.save_freq is not None:
            cmd.append(f"--save_freq={args.save_freq}")
        else:
            cmd.append("--save_freq=10000")
        if args.eval_freq is not None:
            cmd.append(f"--eval_freq={args.eval_freq}")
        else:
            cmd.append("--eval_freq=0")

    return cmd


def get_resolved_output_dir(args: argparse.Namespace) -> Path:
    job_name = args.job_name or default_job_name(args)
    output_dir = args.output_dir or default_output_dir(args, job_name)
    return Path(output_dir)


def main() -> int:
    args = parse_args()
    ensure_lerobot_import_path()

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    log(f"[train-hf] Inspecting dataset metadata for {args.dataset_repo_id} ...")
    meta = LeRobotDatasetMetadata(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        revision=args.dataset_revision,
    )

    camera_keys = meta.camera_keys
    selected_camera_key, top_candidates = choose_camera_key(camera_keys, args.camera_key)

    log(f"[train-hf] Dataset root: {meta.root}")
    log(f"[train-hf] Dataset version: {meta.info.get('codebase_version', 'unknown')}")
    log(f"[train-hf] Camera keys: {camera_keys if camera_keys else '(none)'}")
    log(f"[train-hf] Top-view camera candidates: {top_candidates if top_candidates else '(none found)'}")
    if selected_camera_key is not None:
        log(f"[train-hf] Selected camera key: {selected_camera_key}")
    else:
        log("[train-hf] Selected camera key: (none)")

    if not camera_keys:
        log("[train-hf] ERROR: dataset exposes no image/video keys. This wrapper expects a visual dataset.")
        return 1

    rename_map: dict[str, str] = {}
    if args.rename_camera_to_front:
        if not args.camera_key:
            log("[train-hf] ERROR: --rename-camera-to-front requires --camera-key.")
            return 1
        if args.camera_key not in camera_keys:
            log(f"[train-hf] ERROR: camera key {args.camera_key!r} not found in dataset camera keys {camera_keys}.")
            return 1
        if args.camera_key != "observation.images.front":
            rename_map[args.camera_key] = "observation.images.front"
            log(f"[train-hf] Applying rename_map: {rename_map}")

    if selected_camera_key is not None and not is_top_down_camera_key(selected_camera_key):
        log(
            "[train-hf] WARNING: the selected camera key does not look top-down. "
            "If you specifically want an XY-plane overhead view, use a dataset with a top/overhead camera "
            "or pass --camera-key for the matching stream."
        )

    cmd = build_train_command(args, rename_map)
    output_dir = get_resolved_output_dir(args)
    log_file = output_dir / "train.log"
    pretty_cmd = " ".join(shlex.quote(part) for part in cmd)
    log("[train-hf] Launch command:")
    log(pretty_cmd)
    log(f"[train-hf] Run log: {log_file}")
    log(
        "[train-hf] Metrics watcher:"
        f" python scripts/watch_train_metrics.py --log-file {shlex.quote(str(log_file))}"
    )

    if args.print_only:
        return 0

    env = extend_pythonpath(os.environ.copy(), PROJECT_ROOT, LEROBOT_SRC)
    output_dir.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_handle.write(line)
            log_handle.flush()
        return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
