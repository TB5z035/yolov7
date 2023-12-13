from pathlib import Path
import torch
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import numpy as np
from typing import Any

class DetectModel:
    def __init__(self, model, **kwargs) -> None:
        """Create DetectModel with specific model
        Args:
            model (str): type of model
        """
        self.model: str = model
        raise NotImplementedError
    def infer(self, image: np.ndarray, prompt: Any, **kwargs) -> dict:
        # image shape H，W，C
        # SAM_prompt : numpy.ndarray  ,shape (1,2)
        raise NotImplementedError
        return {
            "bbox": None,  # list[np.ndarray(center_h, center_w, height, width)]
            "text": None,  # str
            "score": None  # list[float | np.ndarray]
        }


class Detector(DetectModel):
    def __init__(self, model, **kwargs):
        self.model=model

    def infer(self, image: str,prompt: Any,  **kwargs) -> dict:
        dict={3:"mug",2:"microwave_door",1:"cabinet_handle",0:"bowl"}
        bbox=[]
        score=[]
        if torch.cuda.is_available():
            device = select_device("0")
        else:
            device = select_device("cpu")
        half = device.type != 'cpu'  # half precision only supported on CUDA
        model = attempt_load(self.model, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(640, s=stride)  # check img_size
        if half:
            model.half()  # to FP16
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        dataset = LoadImages(image, img_size=imgsz, stride=stride)
        path, img, im0s, vid_cap = next(iter(dataset))
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img)[0]
        pred = non_max_suppression(pred, 0.2,0.45, classes=None, agnostic=False)
        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
        det = pred[0]
        p = Path(p)  # to Path
        if len(det):
            text=None
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                cls=int(cls.item())
                if dict[cls]==prompt:
                    text=prompt
                    xyxy=torch.stack(xyxy).cpu().numpy()
                    bbox.append(xyxy)
                    conf = conf.cpu().numpy()
                    score.append(conf)
        return {
            "bbox": bbox,  # list[np.ndarray(center_h, center_w, height, width)]
            "text": text,  # str
            "score": score  # list[float | np.ndarray]
        }
# example
# camera = Detector(r"C:\Users\lsewcx\Desktop\yolo\yolov7-1\last.pt")
# abc=camera.infer(r"C:\Users\lsewcx\Desktop\yolo\yolov7-1\ceshi.png","microwave_door")
# print(abc)



