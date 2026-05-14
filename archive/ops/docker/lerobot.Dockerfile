# VLA Arena — LeRobot + IsaacLab Arena + OpenVLA
# Based on IsaacLab-Arena's Dockerfile.isaaclab_arena, with Isaac Sim 5.1.0
#
# Prerequisites:
#   cd /home/rocket/Projects/IsaacLab-Arena && git submodule update --init --recursive
#
# Build (from IsaacLab-Arena root, with VLA project as build context):
#   docker build -t vla-arena:latest \
#     -f /home/rocket/Projects/vla-world-model-control/docker/lerobot.Dockerfile \
#     /home/rocket/Projects/IsaacLab-Arena
#
# Run:
#   docker run --gpus all --rm -it \
#     -v /home/rocket/Projects/vla-world-model-control:/workspace/vla \
#     -v /home/rocket/Projects/vla-world-model-control/datasets:/workspace/datasets \
#     -v /home/rocket/Projects/vla-world-model-control/outputs:/workspace/outputs \
#     vla-arena:latest
#
# Inside container:
#   cd /workspace/vla && pip install -e .
#   python -c "from isaaclab_arena_vla.environments import SO100PickAndPlaceEnvironment; print('OK')"

ARG BASE_IMAGE=nvcr.io/nvidia/isaac-sim:5.1.0
FROM ${BASE_IMAGE}

ARG WORKDIR="/workspace"
ENV WORKDIR=${WORKDIR}
WORKDIR "${WORKDIR}"

# Hide conflicting Vulkan files
RUN if [ -e "/usr/share/vulkan" ] && [ -e "/etc/vulkan" ]; then \
      mv /usr/share/vulkan /usr/share/vulkan_hidden; \
    fi

# System dependencies
RUN apt-get update && apt-get install -y \
    git git-lfs cmake sudo python3-pip ffmpeg libusb-1.0-0-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

################################
# Install Isaac Lab (from Arena submodule)
################################
COPY ./submodules/IsaacLab ${WORKDIR}/submodules/IsaacLab
ENV ISAACLAB_PATH=${WORKDIR}/submodules/IsaacLab
ENV TERM=xterm

RUN ln -s /isaac-sim/ ${WORKDIR}/submodules/IsaacLab/_isaac_sim
RUN for DIR in ${WORKDIR}/submodules/IsaacLab/source/isaaclab*/; do \
        pip install --no-deps -e "$DIR"; \
    done
RUN chmod 777 -R /isaac-sim/kit/
RUN ${ISAACLAB_PATH}/isaaclab.sh -i

# Patch osqp
RUN if python -c "import qpsolvers; print(qpsolvers.available_solvers)" 2>/dev/null | grep -q "osqp"; then \
        echo "OSQP OK"; \
    else \
        /isaac-sim/python.sh -m pip install qpsolvers==4.8.1 2>/dev/null || true; \
    fi

################################
# Install IsaacLab Arena
################################
COPY isaaclab_arena ${WORKDIR}/isaaclab_arena
COPY isaaclab_arena_g1 ${WORKDIR}/isaaclab_arena_g1
COPY isaaclab_arena_gr00t ${WORKDIR}/isaaclab_arena_gr00t
COPY setup.py pyproject.toml ${WORKDIR}/
RUN /isaac-sim/python.sh -m pip install -e ${WORKDIR}/

# Arena pip deps
RUN /isaac-sim/python.sh -m pip install \
    pytest typing_extensions onnxruntime

################################
# VLA + LeRobot dependencies
################################
RUN /isaac-sim/python.sh -m pip install \
    "transformers>=4.40,<4.50" \
    "timm>=0.9.10,<1.0" \
    bitsandbytes accelerate scipy lerobot

# HuggingFace CLI
RUN pip install huggingface-hub[cli]

################################
# Shell config
################################
RUN echo "alias python='/isaac-sim/python.sh'" >> /etc/bash.bashrc && \
    echo "alias pip3='/isaac-sim/python.sh -m pip'" >> /etc/bash.bashrc && \
    echo "PS1='[VLA Arena] \[\e[0;32m\]\u \[\e[0;34m\]\w\[\e[0m\] \$ '" >> /etc/bash.bashrc && \
    cp /etc/bash.bashrc /root/.bashrc

VOLUME ["/workspace/datasets", "/workspace/outputs"]
ENV HF_HOME=/workspace/.cache/huggingface

CMD ["/bin/bash"]
