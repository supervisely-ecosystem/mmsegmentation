FROM supervisely/base-py-sdk:6.4.3

RUN pip3 install -U torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install mmcv-full==1.4.4 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html -U

RUN pip3 install mmsegmentation==0.21.1
RUN pip3 install pyyaml==6.0

RUN mkdir -p /tmp/mmseg \
    && wget https://github.com/open-mmlab/mmsegmentation/archive/refs/tags/v0.21.1.tar.gz -P /tmp/mmseg \
    && tar -xvf /tmp/mmseg/v0.21.1.tar.gz -C /tmp/mmseg
