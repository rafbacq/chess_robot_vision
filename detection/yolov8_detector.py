# YOLOv8 detection wrapper


from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from ultralytics import YOLO
import 
from IPython.display import display, Image #remember to add this to requirements


from roboflow import Roboflow
#rf = Roboflow(api_key="API KEY")
#project = rf.workspace("models-mc3oh").project("chess-pieces-detection-v4fl1")
#dataset = project.version(1).download("yolov8")
#!yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=25 imgsz=800 plots=True


@dataclass
class Detection:
    cls_name: str
    conf: float
    xyxy: np.ndarray  # (4,) float32


class YoloV8Detector:
    #Ultralytics YOLOv8 wrapper.

    def __init__(self, model_path: str, class_names: Optional[Sequence[str]] = None, device: Optional[str] = None):
        self.model = YOLO(model_path)
        self.class_names = list(class_names) if class_names is not None else None
        self.device = device

    def _class_name(self, cls_id: int) -> str:
        if self.class_names is not None and 0 <= cls_id < len(self.class_names):
            return self.class_names[cls_id]
        try:
            names = self.model.names
            if isinstance(names, dict):
                return names.get(cls_id, str(cls_id))
            if isinstance(names, list):
                return names[cls_id]
        except Exception:
            pass
        return str(cls_id)

    def detect(self, image_bgr: np.ndarray, conf_thres: float = 0.35) -> List[Detection]:
        results = self.model(image_bgr, verbose=False, device=self.device)
        r0 = results[0]
        out: List[Detection] = []

        if r0.boxes is None or len(r0.boxes) == 0:
            return out

        xyxy = r0.boxes.xyxy.cpu().numpy()
        conf = r0.boxes.conf.cpu().numpy()
        cls = r0.boxes.cls.cpu().numpy().astype(int)

        for bb, c, k in zip(xyxy, conf, cls):
            if float(c) < conf_thres:
                continue
            out.append(Detection(self._class_name(int(k)), float(c), bb.astype(np.float32)))
        return out

    @staticmethod
    def annotate(image_bgr: np.ndarray, dets: List[Detection]) -> np.ndarray:
        import cv2
        img = image_bgr.copy()
        for d in dets:
            x1, y1, x2, y2 = map(int, d.xyxy)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{d.cls_name} {d.conf:.2f}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return img