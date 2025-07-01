import importlib

from inferencer.core.inferencer_base import InferencerBase


def is_connected(dummy):
    try:
        return "is_connected", {"state": 1, "is_connected": True}
    except:
        return "exception", {"state": 0}


def get_inferencer(model_name: str, model_args: dict, logger, reload_module=False) -> InferencerBase:
    """module_loader = {function name used in request: [python file name, class name]}"""
    module_loader = {"box_depallet": ["depal_box", "DepalBox"], "tire_hole": ["tire_hole", "TireHole"]}
    if model_name == "is_connected":
        return is_connected
    elif model_name in module_loader.keys():
        submodule_name, class_name = module_loader[model_name]
        target_module = importlib.import_module(f"inferencer.{submodule_name}")
        if reload_module:
            importlib.reload(target_module)
        return getattr(target_module, class_name)(model_name, model_args, logger)
    else:
        raise Exception("Not supported model")
