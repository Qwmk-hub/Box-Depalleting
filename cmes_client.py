import argparse
import json

from cmes_ipc.ipc_grpc import call_grpc_function


def dummy_json():
    json_data = {"temp": "temp"}
    encoded_data = json.dumps(json_data)
    return bytes(encoded_data, "utf-8")


def read_from_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f_json:
        json_data = json.load(f_json)
    # do something if need
    encoded_data = json.dumps(json_data)
    return bytes(encoded_data, "utf-8")


def run_client(func_name):
    if func_name == "is_connected":
        request_data = dummy_json()
    elif func_name == "box_depallet":
        # TODO: change this part if you are in depal project
        request_data = read_from_json("C:/Users/User/Downloads/cmes_hyu_2025/data_samples/depal_box/input_sample.json")

    elif func_name == "tire_hole":
        # TODO: change this part if you are in tire hole project
        request_data = read_from_json("C:/Users/User/Downloads/cmes_hyu_2025/data_samples/tire_holes_detection/input_sample.json")
    else:
        raise KeyError("Not Supported function")

    print(f"request to {func_name} -> {request_data}")
    response_json = call_grpc_function(func_name, request_data)
    print(f"response from {func_name} -> {response_json}")
    return response_json


if __name__ == "__main__":
    """
    python cmes_client.py is_connected
    python cmes_client.py reload_ai
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("function", type=str, default="is_connected")
    parser.add_argument("-t", "--times", type=int, default=0)
    args = parser.parse_args()

    import numpy as np

    from utils import CheckExecTime

    repeat = args.times
    if repeat:
        lst_e = []
        for _ in range(repeat):
            with CheckExecTime() as e:
                run_client(args.function)
            lst_e.append(float(e))

        print(np.mean(lst_e).round(3))
        print(np.array(lst_e).round(3))
    else:
        with CheckExecTime() as e:
            run_client(args.function)
        print(f"{e:.3f}")
