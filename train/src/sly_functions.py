def get_bg_class_name(class_names):
    possible_bg_names = ["background", "bg", "unlabeled", "neutral", "__bg__"]
    for name in class_names:
        if name.lower() in possible_bg_names:
            return name
    return None
