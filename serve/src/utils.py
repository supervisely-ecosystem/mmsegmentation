def filter_models_structure(models: dict):
    filtered_models = []
    for arch_type in models.keys():
        for checkpoint in models[arch_type]["checkpoints"]:
            checkpoint["meta"]["task_type"] = "semantic segmentation"
            checkpoint["meta"]["arch_type"] = arch_type
            checkpoint["meta"]["arch_link"] = models[arch_type]["config_url"]
            filtered_models.append(checkpoint)
    return filtered_models