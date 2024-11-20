import sys

sys.path.insert(0, "../")

from serve.src.mmsegm_model import MMSegmentationModel


class MMSegmentationModelBench(MMSegmentationModel):
    in_train = True
