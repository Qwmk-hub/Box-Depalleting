import cv2
import numpy as np
import torch
from ultralytics import YOLO

from inferencer.core.inferencer_base import InferencerBase
from utils import CheckExecTime


class TireHole(InferencerBase):
    """
    only methods `warmup_model` and `forward` should be overridden
    there are no restrictions on implementing new methods besides the overridden ones
    """

    def __init__(self, model_name, model_args, logger) -> None:
        super().__init__(model_name, model_args, logger)
        self.model = YOLO(model_args["path_weight"])
        if torch.cuda.is_available():
            self.model.to(self.device)
        self.warmup_model()

    def warmup_model(self):
        with CheckExecTime() as elapsed:
            dummy = np.zeros((1080, 1920, 3), dtype=np.uint8)  # TODO: Consider changing the size of the dummy image
            self.model.predict(source=dummy, verbose=False)
        self.logger.debug(f"Model is warmed up in {int(float(elapsed) * 1000)}ms")

    def forward(self, input_data):
        """
        input_data

        model             : "CMES_TIRE_HOLE_CENTER",
        in_img_path       : input color 이미지 절대 경로 ( 3 채널 bgr),
        id                : 이미지 저장 중복 방지를 위한 unique id,
        roi               : "x_small, x_large, y_small, y_large" (좌표계는 이미지 좌측 상단이 원점),
        img_root_dir      : 결과 이미지 저장 절대 경로
        """
        rgb_img = cv2.imread(input_data["in_img_path"])
        # TODO: if you need model, roi, id, img_root_dir, read them first and use them in the rest of algorithm

        with CheckExecTime() as elapsed:
            result = self.model.predict(rgb_img)

        ############### TODO: Change this part to with your own codes #################
        from ultralytics.engine.results import Results

        if isinstance(result[0], Results):
            result = "implement the codes!"
        print(f"result: {result}")
        ###############################################################################

        self.logger.debug(f"Process Time - {int(float(elapsed) * 1000)}ms")
        return result
