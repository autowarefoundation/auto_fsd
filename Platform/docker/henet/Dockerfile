FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG HENET_REVISION=29ca81dd109cabe0a0c53ee354c4a74ad1559740
ARG TORCH_CUDA_ARCH_LIST="8.6+PTX"

ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        libgl1 \
        libglib2.0-0 \
        ninja-build \
        python3.10 \
        python3.10-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3.10 -m pip install --no-cache-dir --upgrade \
        pip==23.3.2 \
        setuptools==69.0.3 \
        wheel==0.41.2

RUN python3.10 -m pip install --no-cache-dir \
        torch==2.0.1+cu118 \
        torchvision==0.15.2+cu118 \
        torchaudio==2.0.2+cu118 \
        -f https://download.pytorch.org/whl/torch_stable.html \
    && python3.10 -m pip install --no-cache-dir \
        mmcv-full==1.6.2 \
        -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html \
    && python3.10 -m pip install --no-cache-dir \
        mmdet==2.28.2 \
        mmsegmentation==0.30.0

RUN python3.10 -m pip install --no-cache-dir \
        aiobotocore==2.17.0 \
        boto3==1.35.93 \
        cloudpickle==2.2.1 \
        einops==0.7.0 \
        flytekit==1.13.15 \
        fsspec==2024.6.1 \
        fvcore==0.1.5.post20221221 \
        iopath==0.1.9 \
        kornia==0.7.1 \
        kubernetes==28.1.0 \
        lyft-dataset-sdk==0.0.8 \
        networkx==2.2 \
        numba==0.58.1 \
        nuscenes-devkit==1.1.11 \
        opencv-python==4.8.1.78 \
        pandas==1.4.4 \
        Pillow==10.1.0 \
        plyfile==1.0.2 \
        pyquaternion==0.9.9 \
        s3fs==2024.6.1 \
        scikit-image==0.19.3 \
        scipy==1.10.1 \
        shapely==1.8.5 \
        tensorboard==2.14.0 \
        timm==0.9.12 \
        trimesh==2.35.39 \
        yacs==0.1.9

RUN git clone https://github.com/VDIGPKU/HENet.git /opt/HENet \
    && git -C /opt/HENet checkout "${HENET_REVISION}" \
    && test "$(git -C /opt/HENet rev-parse HEAD)" = "${HENET_REVISION}" \
    && cd /opt/HENet/mmdet3d/ops/csrc \
    && python3.10 setup.py build_ext --inplace \
    && cd /opt/HENet/mmdet3d/ops/deformattn \
    && python3.10 setup.py build install \
    && cd /opt/HENet \
    && python3.10 setup.py develop \
    && cd /opt/HENet/detr2 \
    && python3.10 setup.py develop

WORKDIR /workspace

COPY Model/ /workspace/Model/
COPY Platform/pipelines/ /workspace/Platform/pipelines/

ENV PYTHONPATH=/opt/HENet:/workspace/Model:/workspace

RUN python3.10 -m pip check \
    && python3.10 -c \
    "import importlib, torch; \
from mmcv import Config; \
from mmdet3d.models import build_model; \
from Platform.pipelines import henet_occupancy, henet_runtime, workflows; \
assert henet_occupancy.HENET_REVISION; \
assert henet_runtime.HENET_INPUT_WIDTH == 1152; \
assert workflows.precompute_henet_occupancy_artifacts.name; \
config=Config.fromfile('/opt/HENet/configs/henet/henet_det_bevseg.py'); \
config.model.pretrained=None; \
config.model.train_cfg=None; \
build_model(config.model, test_cfg=config.get('test_cfg')); \
assert torch.__version__.startswith('2.0.1')"
