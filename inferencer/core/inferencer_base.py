import json
import os
from abc import ABC, abstractmethod

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import torch

from utils import get_time_formatted


class InferencerBase(ABC):
    def __init__(self, model_name, model_args, logger) -> None:
        self.model_name = model_name
        self.model_args = model_args
        self.logger = logger

        device = model_args.get("gpu_id", "cpu")
        if isinstance(device, int) and device >= 0:
            if torch.cuda.is_available():
                device = f"cuda:{device}"
            else:
                self.logger.warning("CUDA is not available")
        else:
            self.logger.warning("CUDA is not set, CPU will be used")
        self.device = device

        timestamp = get_time_formatted(inc_time=True)
        self.path_debug = f"debug/{model_name}/{timestamp}"
        os.makedirs(self.path_debug, exist_ok=True)
        with open(os.path.join(self.path_debug, "args.json"), "w") as f:
            json.dump(model_args, f)

    @abstractmethod
    def forward(self, input_data):
        pass

    def warmup_model(self):
        raise NotImplementedError()

    def __call__(self, input_data) -> dict:
        data_decoded = self._decode(input_data)
        output_data = self.forward(data_decoded)
        return self.model_name, output_data

    def _decode(self, data):
        return json.loads(data)

    def make_result_dir(self, config):
        self.save_result_path = config.get("save_result_path", "")
        self.save_result = False

        if self.save_result_path != "":
            self.save_result_path = self.save_result_path + "/" + get_time_formatted(inc_time=False)
            os.makedirs(self.save_result_path, exist_ok=True)
            self.save_result = True
