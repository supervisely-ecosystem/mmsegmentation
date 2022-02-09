from mmseg.datasets.custom import CustomDataset
from mmseg.datasets.builder import DATASETS


@DATASETS.register_module()
class SuperviselyDataset(CustomDataset):
    CLASSES = None
    PALETTE = None

    def __init__(self, split, classes, palette, img_suffix, seg_map_suffix, ignore_index=255, **kwargs):
        SuperviselyDataset.CLASSES = classes
        SuperviselyDataset.PALETTE = palette
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix,
                         split=split, ignore_index=ignore_index, **kwargs)