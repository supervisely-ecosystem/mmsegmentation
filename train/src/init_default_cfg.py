import sly_globals as g

def init_default_cfg_params(state):
    state["optimizer"] = "SGD"
    state["lr"] = 0.001
    state["weightDecay"] = 0
    state["decodeHeadLoss"] = "CrossEntropyLoss"
    state["auxiliaryHeadLoss"] = "CrossEntropyLoss"
    state["decodeHeadLossWeight"] = 1.0
    state["auxiliaryHeadLossWeight"] = 0.4
    state["lrPolicy"] = "Cyclic"
    state["useWarmup"] = False
    state["warmup"] = "constant"
    state["warmupIters"] = 0
    state["warmupRatio"] = 0.1
    state["schedulerByEpochs"] = False
    state["minLREnabled"] = False
    state["minLR"] = None
    state["minLRRatio"] = None
    state["power"] = 1
    state["momentum"] = 0.9
    state["beta1"] = 0.9
    state["beta2"] = 0.999
    state["input_size"] = {
        "value": {
            "width": 256,
            "height": 256,
            "proportional": False
        },
        "options": {
            "proportions": {
              "width": 100,
              "height": 100
            },
            "min": 64
        }
    }
    state["batchSizePerGPU"] = 4
    state["workersPerGPU"] = 2


def init_default_cfg_args(cfg):
    params = []
    if hasattr(cfg.model, "decode_head") and cfg.model.decode_head is not None:
        decode_is_list = isinstance(cfg.model.decode_head, list)
        if decode_is_list and hasattr(cfg.model.decode_head[0], "loss_decode") or not decode_is_list and hasattr(cfg.model.decode_head, "loss_decode"):
            params.extend([
                {
                    "field": "state.decodeHeadLoss",
                    "payload": cfg.model.decode_head[0].loss_decode.type if decode_is_list else cfg.model.decode_head.loss_decode.type
                },
                {
                    "field": "state.decodeHeadLossWeight",
                    "payload": cfg.model.decode_head[0].loss_decode.loss_weight if decode_is_list else cfg.model.decode_head.loss_decode.loss_weight
                },
            ])
    if hasattr(cfg.model, "auxiliary_head") and cfg.model.auxiliary_head is not None:
        params.extend([
            {
                "field": "state.auxiliaryHeadLoss",
                "payload": cfg.model.auxiliary_head[0].loss_decode.type if isinstance(cfg.model.auxiliary_head,
                                                                                list) else cfg.model.auxiliary_head.loss_decode.type
            },
            {
                "field": "state.auxiliaryHeadLossWeight",
                "payload": cfg.model.auxiliary_head[0].loss_decode.loss_weight if isinstance(cfg.model.auxiliary_head, list) else cfg.model.auxiliary_head.loss_decode.loss_weight
            }
        ])
    if hasattr(cfg.data, "samples_per_gpu"):
        params.extend([{
            "field": "state.batchSizePerGPU",
            "payload": cfg.data.samples_per_gpu
        }])
    if hasattr(cfg.data, "workers_per_gpu"):
        params.extend([{
            "field": "state.workersPerGPU",
            "payload": cfg.data.workers_per_gpu
        }])
    if hasattr(cfg, "crop_size"):
        params.extend([{
            "field": "state.input_size.value.height",
            "payload": cfg.crop_size[0]
        },{
            "field": "state.input_size.value.width",
            "payload": cfg.crop_size[1]
        },{
            "field": "state.input_size.options.proportions.height",
            "payload": 100
        },{
            "field": "state.input_size.options.proportions.width",
            "payload": 100 * (cfg.crop_size[1] / cfg.crop_size[0])
        }])
    if hasattr(cfg.optimizer, "type"):
        params.extend([{
            "field": "state.optimizer",
            "payload": cfg.optimizer.type
        }])
    if hasattr(cfg.optimizer, "lr"):
        params.extend([{
            "field": "state.lr",
            "payload": cfg.optimizer.lr
        }])
    if hasattr(cfg.optimizer, "weight_decay"):
        params.extend([{
            "field": "state.weightDecay",
            "payload": cfg.optimizer.weight_decay
        }])
    if hasattr(cfg.optimizer, "momentum"):
        params.extend([{
            "field": "state.momentum",
            "payload": cfg.optimizer.momentum
        }])
    if hasattr(cfg.optimizer, "betas"):
        params.extend([{
            "field": "state.beta1",
            "payload": cfg.optimizer.betas[0]
        },{
            "field": "state.beta2",
            "payload": cfg.optimizer.betas[1]
        }])
    # take lr scheduler params
    if hasattr(cfg, "lr_config"):
        if hasattr(cfg.lr_config, "policy"):
            policy = cfg.lr_config.policy.capitalize()
            params.extend([{
                "field": "state.lrPolicy",
                "payload": "Cyclic"
            }])
        if hasattr(cfg.lr_config, "warmup"):
            warmup = cfg.lr_config.warmup
            params.extend([{
                "field": "state.useWarmup",
                "payload": warmup is not None
            },{
                "field": "state.warmup",
                "payload": warmup
            }])
        if hasattr(cfg.lr_config, "warmup_iters"):
            warmup_iters = cfg.lr_config.warmup_iters
            # warmup iters no more than half of all data length
            if warmup_iters > g.project_info.items_count * 0.5 // cfg.data.samples_per_gpu:
                warmup_iters = g.project_info.items_count * 0.5 // cfg.data.samples_per_gpu
            params.extend([{
                "field": "state.warmupIters",
                "payload": warmup_iters
            }])
        if hasattr(cfg.lr_config, "warmup_ratio"):
            params.extend([{
                "field": "state.warmupRatio",
                "payload": cfg.lr_config.warmup_ratio
            }])
        if hasattr(cfg.lr_config, "by_epoch"):
            params.extend([{
                "field": "state.schedulerByEpochs",
                "payload": cfg.lr_config.by_epoch
            }])
        if hasattr(cfg.lr_config, "min_lr"):
            params.extend([{
                "field": "state.minLREnabled",
                "payload": True
            },{
                "field": "state.minLR",
                "payload": cfg.lr_config.min_lr
            }])
        if hasattr(cfg.lr_config, "power"):
            params.extend([{
                "field": "state.power",
                "payload": cfg.lr_config.power
            }])

    return params