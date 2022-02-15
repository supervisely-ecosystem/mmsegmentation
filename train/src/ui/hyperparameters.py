import os
import supervisely as sly
import sly_globals as g


def init_general(state):
    state["epochs"] = 15
    state["gpusId"] = 0

    state["valInterval"] = 1
    state["logConfigInterval"] = 5

def init_checkpoints(state):
    state["checkpointInterval"] = 1
    state["maxKeepCkptsEnabled"] = True
    state["maxKeepCkpts"] = 2
    state["saveLast"] = True
    state["saveBest"] = True

def init_optimizer(state):
    state["nesterov"] = False
    state["amsgrad"] = False
    state["momentumDecay"] = 0.004
    state["gradClipEnabled"] = False
    state["maxNorm"] = 1

def init_losses(data, state):
    data["availableLosses"] = ["CrossEntropyLoss", "DiceLoss", "FocalLoss", "LovaszLoss"]
    state["useClassWeights"] = False
    state["classWeights"] = ""
    data["classesList"] = [class_obj["title"] for class_obj in g.project_meta.obj_classes.to_json()]
    state["decodeSmoothLoss"] = 1
    state["decodeExpLoss"] = 2
    state["auxiliarySmoothLoss"] = 1
    state["auxiliaryExpLoss"] = 2
    state["decodeAlpha"] = 0.5
    state["decodeGamma"] = 2.0
    state["auxiliaryAlpha"] = 0.5
    state["auxiliaryGamma"] = 2.0
    data["availableMetrics"] = ["mIoU", "mDice"]
    state["evalMetrics"] = ["mIoU", "mDice"]


def init_lr_scheduler(data, state):
    # LR scheduler params
    data["availableLrPolicy"] = ["Fixed", "Step", "Exp", "Poly", "Inv", "CosineAnnealing", "FlatCosineAnnealing",
                                 "CosineRestart", "Cyclic", "OneCycle"]
    data["fullPolicyNames"] = ["Constant LR", "Step LR", "Exponential LR", "Polynomial LR Decay",
                               "Inverse Square Root LR", "Cosine Annealing LR", "Flat + Cosine Annealing LR",
                               "Cosine Annealing with Restarts", "Cyclic LR", "OneCycle LR"]

    state["lr_step"] = ""
    state["gamma"] = 0.1
    state["startPercent"] = 0.75
    state["periods"] = ""
    state["restartWeights"] = ""
    state["highestLRRatio"] = 10
    state["lowestLRRatio"] = 1e-3
    state["cyclicTimes"] = 10
    state["stepRatioUp"] = 0.4
    state["annealStrategy"] = "cos"
    state["cyclicGamma"] = 1
    state["totalStepsEnabled"] = False
    state["totalSteps"] = None
    state["maxLR"] = ""
    state["pctStart"] = 0.3
    state["divFactor"] = 25
    state["finalDivFactor"] = 1e4
    state["threePhase"] = False
    state["warmupByEpoch"] = False

def init(data, state):
    init_general(state)
    init_checkpoints(state)
    init_optimizer(state)
    init_losses(data, state)
    init_lr_scheduler(data, state)

    state["currentTab"] = "general"
    state["collapsedWarmup"] = True
    state["collapsed6"] = True
    state["disabled6"] = True
    state["done6"] = False


def restart(data, state):
    data["done6"] = False


@g.my_app.callback("use_hyp")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def use_hyp(api: sly.Api, task_id, context, state, app_logger):
    g.evalMetrics = state["evalMetrics"]
    fields = [
        {"field": "data.done6", "payload": True},
        {"field": "state.collapsed7", "payload": False},
        {"field": "state.disabled7", "payload": False},
        {"field": "state.activeStep", "payload": 7},
    ]
    g.api.app.set_fields(g.task_id, fields)