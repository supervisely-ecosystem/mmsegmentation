# ============================================================================
# STAGE 1: Builder Stage
# ============================================================================

# todo: remove; temp
# ARG tag_ref_name=6.73.531
ARG tag_ref_name=0.0.1

FROM python:3.10-slim-bookworm AS builder

ENV LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
  ca-certificates \
  wget \
  && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install pip==26.0.1
RUN pip install setuptools==78.1.1 wheel==0.46.2

RUN pip install -U torch==1.10.0+cu111 torchvision==0.11.0+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install mmcv-full==1.4.4 \
  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html -U
# RUN pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12/index.html -U

# RUN pip install mmcv-full==1.7.0
RUN pip install mmsegmentation==0.23.0
RUN pip install mmcls==0.21.0
RUN pip install pyyaml==6.0
RUN pip install yapf==0.40.1

# RUN pip install torch==1.13.0
# RUN pip install torchvision==0.14.0

# COPY dev_requirements.txt dev_requirements.txt
# RUN pip install -r dev_requirements.txt

RUN pip install setuptools==69.5.1

RUN pip install openmim
RUN pip install ffmpeg-python==0.2.0
RUN pip install pyyaml==6.0

RUN mkdir -p /tmp/mmseg \
  && wget https://github.com/open-mmlab/mmsegmentation/archive/refs/tags/v0.23.0.tar.gz \
    -P /tmp/mmseg \
  && tar -xvf /tmp/mmseg/v0.23.0.tar.gz -C /tmp/mmseg

# todo: remove; temp
# RUN pip install supervisely[model-benchmark]==$tag_ref_name
RUN pip install git+https://github.com/supervisely/supervisely.git@hardened-docker-images

# ============================================================================
# STAGE 2: Runtime Stage (Hardened)
# ============================================================================
FROM supervisely/py-sdk-hardened:${tag_ref_name} AS runtime

# Copy Python environment and sources from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /tmp/mmseg /tmp/mmseg

ENV PATH="/opt/venv/bin:$PATH"

LABEL python_sdk_version="$tag_ref_name"
