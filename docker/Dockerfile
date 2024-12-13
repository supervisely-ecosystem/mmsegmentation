FROM supervisely/base-py-sdk:6.72.66

RUN pip3 install -U torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install mmcv-full==1.4.4 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html -U
# RUN pip3 install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12/index.html -U

# RUN pip3 install mmcv-full==1.7.0
RUN pip3 install mmsegmentation==0.23.0
RUN pip3 install mmcls==0.21.0
RUN pip3 install pyyaml==6.0
RUN pip3 install yapf==0.40.1

# RUN pip3 install torch==1.13.0
# RUN pip3 install torchvision==0.14.0

# COPY dev_requirements.txt dev_requirements.txt
# RUN pip3 install -r dev_requirements.txt

RUN pip3 install setuptools==69.5.1

RUN pip3 install openmim
RUN pip3 install ffmpeg-python==0.2.0
RUN pip3 install pyyaml==6.0

RUN mkdir -p /tmp/mmseg \
&& wget https://github.com/open-mmlab/mmsegmentation/archive/refs/tags/v0.23.0.tar.gz -P /tmp/mmseg \
&& tar -xvf /tmp/mmseg/v0.23.0.tar.gz -C /tmp/mmseg

RUN pip3 install supervisely==6.73.242
LABEL python_sdk_version=6.73.242
