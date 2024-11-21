import architectures
import augs
import splits
import os
import sly_globals as g
from mmcv import ConfigDict
from mmseg.apis import set_random_seed


def init_class_weights(state, classes, default_bg_class_weight=0.3):
    if state["useClassWeights"]:
        return [float(weight) for weight in state["classWeights"].split(",")]
    else:
        return [1] * (len(classes) - 1) + [default_bg_class_weight]


def init_cfg_decode_head(cfg, state, classes, class_weights, ind=0):
    head = dict(
        num_classes=len(classes),
        loss_decode=dict(
            type=state["decodeHeadLoss"],
            loss_weight=float(state["decodeHeadLossWeight"]),
            class_weight=class_weights,
        ),
    )
    if cfg.pretrained_model != "PointRend" or ind == 0:
        head["norm_cfg"] = cfg.norm_cfg
    if state["decodeHeadLoss"] == "DiceLoss":
        head["loss_decode"]["smooth"] = state["decodeSmoothLoss"]
        head["loss_decode"]["exponent"] = state["decodeExpLoss"]
    elif state["decodeHeadLoss"] == "FocalLoss":
        head["loss_decode"]["alpha"] = float(state["decodeAlpha"])
        head["loss_decode"]["gamma"] = float(state["decodeGamma"])
    elif state["decodeHeadLoss"] == "LovaszLoss":
        head["loss_decode"]["reduction"] = "none"
    return head


def init_cfg_auxiliary_head(cfg, state, classes, class_weights):
    head = dict(
        norm_cfg=cfg.norm_cfg,
        num_classes=len(classes),
        loss_decode=dict(
            type=state["auxiliaryHeadLoss"],
            loss_weight=float(state["auxiliaryHeadLossWeight"]),
            class_weight=class_weights,
        ),
    )
    if state["auxiliaryHeadLoss"] == "DiceLoss":
        head["loss_decode"]["smooth"] = state["auxiliarySmoothLoss"]
        head["loss_decode"]["exponent"] = state["auxiliaryExpLoss"]
    elif state["auxiliaryHeadLoss"] == "FocalLoss":
        head["loss_decode"]["alpha"] = float(state["auxiliaryAlpha"])
        head["loss_decode"]["gamma"] = float(state["auxiliaryGamma"])
    elif state["auxiliaryHeadLoss"] == "LovaszLoss":
        head["loss_decode"]["reduction"] = "none"
    return head


def init_cfg_optimizer(cfg, state):
    cfg.optimizer.type = state["optimizer"]
    cfg.optimizer.lr = state["lr"]
    cfg.optimizer.weight_decay = state["weightDecay"]
    if state["gradClipEnabled"]:
        cfg.optimizer_config = dict(grad_clip=dict(max_norm=state["maxNorm"], norm_type=2))
    if hasattr(cfg.optimizer, "eps"):
        delattr(cfg.optimizer, "eps")

    if state["optimizer"] == "SGD":
        if hasattr(cfg.optimizer, "betas"):
            delattr(cfg.optimizer, "betas")
        cfg.optimizer.momentum = state["momentum"]
        cfg.optimizer.nesterov = state["nesterov"]
    elif state["optimizer"] in ["Adam", "Adamax", "AdamW", "NAdam", "RAdam"]:
        if hasattr(cfg.optimizer, "momentum"):
            delattr(cfg.optimizer, "momentum")
        if hasattr(cfg.optimizer, "nesterov"):
            delattr(cfg.optimizer, "nesterov")
        cfg.optimizer.betas = (state["beta1"], state["beta2"])
        if state["optimizer"] in ["Adam", "AdamW"]:
            cfg.optimizer.amsgrad = state["amsgrad"]
        if state["optimizer"] == "NAdam":
            cfg.optimizer.momentum_decay = state["momentumDecay"]


def init_cfg_pipelines(cfg):

    train_steps_to_remove = ["RandomFlip", "PhotoMetricDistortion"]
    train_pipeline = []

    def process_step(config_step, train_pipeline):
        if config_step["type"] in train_steps_to_remove:
            return
        elif config_step["type"] == "LoadAnnotations":
            config_step["reduce_zero_label"] = False
            train_pipeline.append(config_step)
            train_pipeline.append(dict(type="SlyImgAugs", config_path=augs.augs_config_path))
            return
        elif config_step["type"] == "Resize":
            if any([x < y for x, y in zip(config_step["img_scale"][:2], cfg.crop_size[:2])]):
                config_step["img_scale"] = cfg.crop_size
        elif config_step["type"] == "Normalize":
            train_pipeline.append(dict(type="Normalize", **cfg.img_norm_cfg))
            return
        elif config_step["type"] in ["RandomCrop", "Pad"]:
            config_step["crop_size" if config_step["type"] == "RandomCrop" else "size"] = (
                cfg.crop_size
            )
        elif config_step["type"] == "Collect":
            config_step["meta_keys"] = (
                "filename",
                "ori_filename",
                "ori_shape",
                "img_shape",
                "scale_factor",
                "img_norm_cfg",
            )

        train_pipeline.append(config_step)

    if hasattr(cfg.data.train, "dataset") and "pipeline" in cfg.data.train.dataset:
        for config_step in cfg.data.train.dataset.pipeline:
            process_step(config_step, train_pipeline)
    elif "pipeline" in cfg.data.train:
        for config_step in cfg.data.train.pipeline:
            process_step(config_step, train_pipeline)

    cfg.train_pipeline = train_pipeline

    test_pipeline = cfg.data.test.pipeline
    for config_step in test_pipeline:
        if config_step["type"] == "MultiScaleFlipAug":
            if (
                config_step["img_scale"][0] < cfg.crop_size[0]
                or config_step["img_scale"][1] < cfg.crop_size[1]
            ):
                config_step["img_scale"] = cfg.crop_size
            transform_pipeline = []
            for transform_step in config_step["transforms"]:
                if transform_step["type"] == "Normalize":
                    transform_pipeline.append(dict(type="Normalize", **cfg.img_norm_cfg))
                    continue
                elif transform_step["type"] == "Resize":
                    transform_step["keep_ratio"] = False
                transform_pipeline.append(transform_step)
            config_step["transforms"] = transform_pipeline

    cfg.val_pipeline = test_pipeline
    cfg.test_pipeline = test_pipeline


def init_cfg_splits(cfg, img_dir, ann_dir, classes, palette):
    cfg.data.train.type = cfg.dataset_type
    cfg.data.train.data_root = cfg.data_root
    cfg.data.train.img_dir = img_dir
    cfg.data.train.ann_dir = ann_dir
    cfg.data.train.pipeline = cfg.train_pipeline
    cfg.data.train.split = splits.train_set_path
    cfg.data.train.classes = classes
    cfg.data.train.palette = palette
    cfg.data.train.img_suffix = ""
    cfg.data.train.seg_map_suffix = ".png"
    if hasattr(cfg.data.train, "times"):
        delattr(cfg.data.train, "times")
    if hasattr(cfg.data.train, "dataset"):
        delattr(cfg.data.train, "dataset")

    cfg.data.val.type = cfg.dataset_type
    cfg.data.val.data_root = cfg.data_root
    cfg.data.val.img_dir = img_dir
    cfg.data.val.ann_dir = ann_dir
    cfg.data.val.pipeline = cfg.val_pipeline
    cfg.data.val.split = splits.val_set_path
    cfg.data.val.classes = classes
    cfg.data.val.palette = palette
    cfg.data.val.img_suffix = ""
    cfg.data.val.seg_map_suffix = ".png"

    cfg.data.test.type = cfg.dataset_type
    cfg.data.test.data_root = cfg.data_root
    cfg.data.test.img_dir = img_dir
    cfg.data.test.ann_dir = ann_dir
    cfg.data.test.pipeline = cfg.test_pipeline
    cfg.data.test.split = None
    cfg.data.test.classes = classes
    cfg.data.test.palette = palette
    cfg.data.test.img_suffix = ""
    cfg.data.test.seg_map_suffix = ".png"


def init_cfg_training(cfg, state):
    cfg.dataset_type = "SuperviselyDataset"
    cfg.data_root = g.project_seg_dir

    cfg.data.samples_per_gpu = state["batchSizePerGPU"]
    cfg.data.workers_per_gpu = state["workersPerGPU"]
    cfg.data.persistent_workers = True

    # TODO: sync with state["gpusId"] if it will be needed
    cfg.gpu_ids = [state["selectedDevice"]]
    # cfg.gpu_ids = range(1)

    cfg.img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
    )
    cfg.crop_size = (state["input_size"]["value"]["width"], state["input_size"]["value"]["height"])
    cfg.load_from = g.local_weights_path

    cfg.work_dir = g.my_app.data_dir

    if not hasattr(cfg, "runner"):
        cfg.runner = ConfigDict()
    cfg.runner.type = "EpochBasedRunner"
    cfg.runner.max_epochs = state["epochs"]
    if hasattr(cfg.runner, "max_iters"):
        delattr(cfg.runner, "max_iters")

    cfg.log_config.interval = state["logConfigInterval"]
    cfg.log_config.hooks = [dict(type="SuperviselyLoggerHook", by_epoch=False)]


def init_cfg_eval(cfg, state):
    cfg.evaluation.interval = state["valInterval"]
    cfg.evaluation.metric = state["evalMetrics"]
    cfg.evaluation.save_best = "auto" if state["saveBest"] else None
    cfg.evaluation.rule = "greater"
    cfg.evaluation.out_dir = g.checkpoints_dir
    cfg.evaluation.by_epoch = True


def init_cfg_checkpoint(cfg, state, classes, palette):
    cfg.checkpoint_config.interval = state["checkpointInterval"]
    cfg.checkpoint_config.by_epoch = True
    cfg.checkpoint_config.max_keep_ckpts = (
        state["maxKeepCkpts"] if state["maxKeepCkptsEnabled"] else 0
    )
    cfg.checkpoint_config.save_last = state["saveLast"]
    cfg.checkpoint_config.out_dir = g.checkpoints_dir
    cfg.checkpoint_config.meta = dict(
        # mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
        # config=cfg.pretty_text,
        CLASSES=classes,
        PALETTE=palette,
    )


def init_cfg_lr(cfg, state):
    lr_config = dict(
        policy=state["lrPolicy"],
        by_epoch=state["schedulerByEpochs"],
        warmup=state["warmup"] if state["useWarmup"] else None,
        warmup_by_epoch=state["warmupByEpoch"],
        warmup_iters=state["warmupIters"],
        warmup_ratio=state["warmupRatio"],
    )
    if state["lrPolicy"] == "Step":
        steps = [int(step) for step in state["lr_step"].split(",")]
        assert len(steps)
        lr_config["step"] = steps
        lr_config["gamma"] = state["gamma"]
        lr_config["min_lr"] = state["minLR"]
    elif state["lrPolicy"] == "Exp":
        lr_config["gamma"] = state["gamma"]
    elif state["lrPolicy"] == "Poly":
        lr_config["min_lr"] = state["minLR"]
        lr_config["power"] = state["power"]
    elif state["lrPolicy"] == "Inv":
        lr_config["gamma"] = state["gamma"]
        lr_config["power"] = state["power"]
    elif state["lrPolicy"] == "CosineAnnealing":
        lr_config["min_lr"] = state["minLR"] if state["minLREnabled"] else None
        lr_config["min_lr_ratio"] = state["minLRRatio"] if not state["minLREnabled"] else None
    elif state["lrPolicy"] == "FlatCosineAnnealing":
        lr_config["min_lr"] = state["minLR"] if state["minLREnabled"] else None
        lr_config["min_lr_ratio"] = state["minLRRatio"] if not state["minLREnabled"] else None
        lr_config["start_percent"] = state["startPercent"]
    elif state["lrPolicy"] == "CosineRestart":
        lr_config["min_lr"] = state["minLR"] if state["minLREnabled"] else None
        lr_config["min_lr_ratio"] = state["minLRRatio"] if not state["minLREnabled"] else None
        lr_config["periods"] = [int(period) for period in state["periods"].split(",")]
        lr_config["restart_weights"] = [
            float(weight) for weight in state["restartWeights"].split(",")
        ]
    elif state["lrPolicy"] == "Cyclic":
        lr_config["target_ratio"] = (state["highestLRRatio"], state["lowestLRRatio"])
        lr_config["cyclic_times"] = state["cyclicTimes"]
        lr_config["step_ratio_up"] = state["stepRatioUp"]
        lr_config["anneal_strategy"] = state["annealStrategy"]
        lr_config["gamma"] = state["cyclicGamma"]
    elif state["lrPolicy"] == "OneCycle":
        lr_config["anneal_strategy"] = state["annealStrategy"]
        lr_config["max_lr"] = [float(maxlr) for maxlr in state["maxLR"].split(",")]
        lr_config["total_steps"] = state["totalSteps"] if state["totalStepsEnabled"] else None
        lr_config["pct_start"] = state["pctStart"]
        lr_config["div_factor"] = state["divFactor"]
        lr_config["final_div_factor"] = state["finalDivFactor"]
        lr_config["three_phase"] = state["threePhase"]
    cfg.lr_config = lr_config


def init_cfg_model(cfg, state, classes):
    class_weights = init_class_weights(state, classes)
    if hasattr(cfg.model.backbone, "pretrained"):
        delattr(cfg.model.backbone, "pretrained")

    if isinstance(cfg.model.decode_head, list):
        for i in range(len(cfg.model.decode_head)):
            head = init_cfg_decode_head(cfg, state, classes, class_weights, ind=i)
            for key in head:
                cfg.model.decode_head[i][key] = head[key]
    else:
        if cfg.pretrained_model == "KNet":
            for i in range(len(cfg.model.decode_head.kernel_update_head)):
                cfg.model.decode_head.kernel_update_head[i].num_classes = len(classes)
            cfg.model.decode_head.kernel_generate_head.num_classes = len(classes)
        else:
            head = init_cfg_decode_head(cfg, state, classes, class_weights)
            for key in head:
                cfg.model.decode_head[key] = head[key]

    if state["useAuxiliaryHead"]:
        if isinstance(cfg.model.auxiliary_head, list):
            for i in range(len(cfg.model.auxiliary_head)):
                head = init_cfg_auxiliary_head(cfg, state, classes, class_weights)
                for key in head:
                    cfg.model.auxiliary_head[i][key] = head[key]
        else:
            head = init_cfg_auxiliary_head(cfg, state, classes, class_weights)
            for key in head:
                cfg.model.auxiliary_head[key] = head[key]


def init_cfg(state, img_dir, ann_dir, classes, palette):
    cfg = architectures.cfg

    init_cfg_model(cfg, state, classes)
    init_cfg_optimizer(cfg, state)
    init_cfg_training(cfg, state)
    init_cfg_pipelines(cfg)
    init_cfg_splits(cfg, img_dir, ann_dir, classes, palette)
    init_cfg_eval(cfg, state)
    init_cfg_checkpoint(cfg, state, classes, palette)
    init_cfg_lr(cfg, state)

    # Set seed to facitate reproducing the result
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    return cfg
