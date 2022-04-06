import supervisely as sly
from mmseg.datasets.builder import PIPELINES
from supervisely.sly_logger import logger
import imgaug.augmenters as iaa


def get_function(category_name, aug_name):
    try:
        submodule = getattr(iaa, category_name)
        aug_f = getattr(submodule, aug_name)
        return aug_f
    except Exception as e:
        logger.error(repr(e))
        # raise e
        return None

def build_pipeline(aug_infos, random_order=False):
    pipeline = []
    for aug_info in aug_infos:
        category_name = aug_info["category"]
        aug_name = aug_info["name"]
        params = aug_info["params"]
        for param_name, param_val in params.items():
            if isinstance(param_val, dict):
                if "x" in param_val.keys() and "y" in param_val.keys():
                    param_val["x"] = tuple(param_val["x"])
                    param_val["y"] = tuple(param_val["y"])
            elif isinstance(param_val, list):
                params[param_name] = tuple(param_val)

        aug_func = get_function(category_name, aug_name)

        aug = aug_func(**params)

        sometimes = aug_info.get("sometimes", None)
        if sometimes is not None:
            aug = iaa.meta.Sometimes(sometimes, aug)
        pipeline.append(aug)
    augs = iaa.Sequential(pipeline, random_order=random_order)
    return augs


def aug_to_python(aug_info):
    pstr = ""
    for name, value in aug_info["params"].items():
        v = value
        if type(v) is list:  #name != 'nb_iterations' and
            v = (v[0], v[1])
        elif type(v) is dict and "x" in v.keys() and "y" in v.keys():
            v = {"x": (v["x"][0], v["x"][1]), "y": (v["y"][0], v["y"][1])}

        if type(value) is str:
            pstr += f"{name}='{v}', "
        else:
            pstr += f"{name}={v}, "
    method_py = f"iaa.{aug_info['category']}.{aug_info['name']}({pstr[:-2]})"

    res = method_py
    if "sometimes" in aug_info:
        res = f"iaa.Sometimes({aug_info['sometimes']}, {method_py})"
    return res


def pipeline_to_python(aug_infos, random_order=False):
    template = \
"""import imgaug.augmenters as iaa
seq = iaa.Sequential([
{}
], random_order={})
"""
    py_lines = []
    for info in aug_infos:
        line = aug_to_python(info)
        _validate = info["python"]
        if line != _validate:
            raise ValueError("Generated python line differs from the one from config: \n\n{!r}\n\n{!r}"
                             .format(line, _validate))
        py_lines.append(line)
    res = template.format('\t' + ',\n\t'.join(py_lines), random_order)
    return res

@PIPELINES.register_module()
class SlyImgAugs(object):
    def __init__(self, config_path):
        self.config_path = config_path
        if self.config_path is not None:
            config = sly.json.load_json_file(self.config_path)
            self.augs = sly.imgaug_utils.build_pipeline(config["pipeline"], random_order=config["random_order"])
            

    def _apply_augs(self, results):
        if self.config_path is None:
            return
        img = results["img"]
        mask = results["gt_semantic_seg"]
        res_img, res_mask = sly.imgaug_utils.apply_to_image_and_mask(self.augs, img, mask)
        results["img"] = res_img
        results["gt_semantic_seg"] = res_mask
        results['img_shape'] = res_img.shape

    def __call__(self, results):
        self._apply_augs(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(config_path={self.config_path})'
        return repr_str