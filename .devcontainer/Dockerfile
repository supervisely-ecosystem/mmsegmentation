FROM supervisely/base-py-sdk:6.72.216

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install python3-pip -y
RUN apt-get install -y git

RUN python3 -m pip install --upgrade pip
RUN pip3 install -U torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install mmcv-full==1.4.4 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html -U

RUN pip3 install mmsegmentation==0.23.0
RUN pip3 install mmcls==0.21.0
RUN pip3 install pyyaml==6.0

RUN pip3 install ffmpeg-python==0.2.0

RUN pip3 install supervisely[model-benchmark]==6.73.274

RUN mkdir -p /tmp/mmseg \
    && wget https://github.com/open-mmlab/mmsegmentation/archive/refs/tags/v0.23.0.tar.gz -P /tmp/mmseg \
    && tar -xvf /tmp/mmseg/v0.23.0.tar.gz -C /tmp/mmseg

RUN apt-get -y install curl

RUN apt -y install wireguard iproute2
RUN apt-get -y install wget