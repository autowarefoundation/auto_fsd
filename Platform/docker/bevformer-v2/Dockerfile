FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG BEVFORMER_REVISION=66b65f3a1f58caf0507cb2a971b9c0e7f842376c
ARG DETECTRON2_REVISION=d1e04565d3bec8719335b88be9e9b961bf3ec464
ARG MMDET3D_REVISION=f1107977dfd26155fc1f83779ee6535d2468f449

ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        libgl1 \
        libglib2.0-0 \
        ninja-build \
        python3.8 \
        python3.8-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3.8 -m pip install --no-cache-dir --upgrade \
        pip==23.3.2 \
        setuptools==59.5.0 \
        wheel==0.42.0

RUN python3.8 -m pip install --no-cache-dir \
        torch==1.9.1+cu111 \
        torchvision==0.10.1+cu111 \
        torchaudio==0.9.1 \
        -f https://download.pytorch.org/whl/torch_stable.html \
    && python3.8 -m pip install --no-cache-dir \
        opencv-python==4.5.5.64 \
    && python3.8 -m pip install --no-cache-dir \
        mmcv-full==1.4.0 \
        -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html \
    && python3.8 -m pip install --no-cache-dir \
        mmdet==2.14.0 \
        mmsegmentation==0.14.1

# CUDA 11.1 cannot emit sm_89 directly; PTX compiled for 8.6 forward-JITs on
# the L40S GPUs used by the g6e node pools.
ARG TORCH_CUDA_ARCH_LIST="8.6+PTX"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

RUN git clone https://github.com/open-mmlab/mmdetection3d.git /opt/mmdetection3d \
    && git -C /opt/mmdetection3d checkout "${MMDET3D_REVISION}" \
    && python3.8 -m pip install --no-cache-dir --no-deps /opt/mmdetection3d

RUN git clone https://github.com/facebookresearch/detectron2.git /opt/detectron2 \
    && git -C /opt/detectron2 checkout "${DETECTRON2_REVISION}" \
    && python3.8 -m pip install --no-cache-dir --no-deps /opt/detectron2

# mmdetection3d and Detectron2 are source-installed with --no-deps above.
# Keep their runtime requirements explicit alongside Flyte's S3 dependency set.
RUN python3.8 -m pip install --no-cache-dir \
        aiobotocore==2.13.3 \
        black==21.4b2 \
        boto3==1.34.162 \
        cloudpickle==2.2.1 \
        einops==0.6.1 \
        flytekit==1.13.15 \
        fsspec==2024.6.1 \
        future==1.0.0 \
        fvcore==0.1.5.post20221221 \
        hydra-core==1.1.2 \
        iopath==0.1.9 \
        kubernetes==28.1.0 \
        lyft-dataset-sdk==0.0.8 \
        matplotlib==3.5.2 \
        networkx==2.2 \
        numba==0.48.0 \
        numpy==1.19.5 \
        nuscenes-devkit==1.1.9 \
        omegaconf==2.1.2 \
        pandas==1.4.4 \
        Pillow==9.5.0 \
        plyfile==0.7.4 \
        pydot==1.4.2 \
        pycocotools==2.0.7 \
        pyquaternion==0.9.9 \
        s3fs==2024.6.1 \
        scikit-image==0.19.3 \
        scipy==1.9.3 \
        seaborn==0.12.2 \
        tabulate==0.9.0 \
        terminaltables==3.1.10 \
        tensorboard==2.13.0 \
        termcolor==2.4.0 \
        timm==0.6.13 \
        tqdm==4.66.5 \
        trimesh==2.35.39 \
        typing-extensions==4.5.0 \
        yacs==0.1.8

# Remove an unused Tk constant import that breaks the headless runtime.
RUN git clone https://github.com/fundamentalvision/BEVFormer.git /opt/BEVFormer \
    && git -C /opt/BEVFormer checkout "${BEVFORMER_REVISION}" \
    && test "$(git -C /opt/BEVFormer rev-parse HEAD)" = "${BEVFORMER_REVISION}" \
    && sed -i '/from tkinter\.messagebox import NO/d' \
        /opt/BEVFormer/projects/mmdet3d_plugin/bevformer/detectors/bevformer_fp16.py \
    && ! grep -Fq "from tkinter.messagebox import NO" \
        /opt/BEVFormer/projects/mmdet3d_plugin/bevformer/detectors/bevformer_fp16.py

WORKDIR /workspace

COPY Model/ /workspace/Model/
COPY Platform/pipelines/ /workspace/Platform/pipelines/

ENV PYTHONPATH=/opt/BEVFormer:/workspace/Model:/workspace

RUN python3.8 -m pip check \
    && python3.8 -c \
    "import cv2, importlib, mmcv, mmdet, mmseg, mmdet3d, torch; \
from Platform.pipelines import bevformer_v2_occupancy, bevformer_v2_runtime, workflows; \
from mmcv import Config; \
from mmdet3d.models import build_model; \
assert bevformer_v2_occupancy.BEVFORMER_V2_WEIGHT_SHA256; \
assert bevformer_v2_runtime.BEVFORMER_V2_IMAGE_WIDTH == 640; \
assert workflows.precompute_bevformer_v2_occupancy_artifacts.name; \
assert torch.__version__.startswith('1.9.1'); \
assert cv2.__version__ == '4.5.5'; \
assert mmcv.__version__ == '1.4.0'; \
assert mmdet.__version__ == '2.14.0'; \
assert mmseg.__version__ == '0.14.1'; \
importlib.import_module('projects.mmdet3d_plugin'); \
config=Config.fromfile('/opt/BEVFormer/projects/configs/bevformerv2/bevformerv2-r50-t8-24ep.py'); \
config.model.pretrained=None; \
config.model.train_cfg=None; \
build_model(config.model, test_cfg=config.get('test_cfg'))"
