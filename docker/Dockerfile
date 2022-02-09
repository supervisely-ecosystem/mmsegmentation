FROM supervisely/base-py-sdk:6.4.3

RUN pip3 install -U torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install mmcv-full==1.4.3 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html -U

RUN pip3 install mmsegmentation==0.20.2
RUN pip3 install pyyaml==6.0
