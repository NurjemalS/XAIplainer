import os
import yaml
import importlib

CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_FILE_PATH, "r") as config_file:
    try:
        config = yaml.safe_load(config_file)
    except yaml.YAMLError as exc:
        print(exc)

def get_model_names():
    return list(config["models"].keys())

def get_model_config(model_name):
    return config["models"][model_name]

def get_model_widget_options(model_name):
    model_config = get_model_config(model_name)
    return model_config.get("widget_options", {})

def load_class(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def load_model(model_name, model_args):
    model_config = get_model_config(model_name)

    model_class =load_class(
        model_config["module_name"],
        model_name
    )

    return model_class(**model_args)

def load_explainer_and_plotter(explainer_name):
    explainer_class = load_class("xai.Explainers", explainer_name)
    plotter_class   = load_class("xai.Plotters", config["explainers"][explainer_name]["plotter"])
    return explainer_class(), plotter_class()

def get_available_explainers_for_model(model_name):
    explainers = list()
    for explainer in config["explainers"]:
        explainer_white_list = config["explainers"][explainer]["model_class_white_list"]
        if explainer_white_list in ["all", config["models"][model_name]["model_class"]]:
            explainers.append(explainer)
    return explainers