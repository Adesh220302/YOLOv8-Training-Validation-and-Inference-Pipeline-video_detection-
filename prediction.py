from ultralytics import YOLO

class YOLOv8Trainer:
    def __init__(self, parameters, model_path):
        self.parameters = parameters
        self.model = YOLO(model_path)

    def show_model_info(self):
        self.model.info()

    def train(self, data_yaml):
        results = self.model.train(
            data=data_yaml,
            epochs=self.parameters.get("epoch", 500),
            imgsz=self.parameters.get("imgsz", 640)
        )
        return results


class YOLOv8Validator:
    def __init__(self, parameters, model_path):
        self.parameters = parameters
        self.model = YOLO(model_path)

    def validate(self, data_yaml):
        metrics = self.model.val(
            data=data_yaml,
            imgsz=self.parameters.get("imgsz", 640),
            batch=self.parameters.get("batch", 16),
            conf=self.parameters.get("conf", 0.5),
            iou=self.parameters.get("iou", 0.4),
            device=self.parameters.get("device", 0)
        )
        return metrics


class YOLOv8Tracker:
    def __init__(self, parameters, model_path):
        self.parameters = parameters
        self.model = YOLO(model_path)

    def track_video(self, video_path):
        source = video_path if video_path else self.parameters.get("source", 0)
        results = self.model.track(
            source=source,
            show=self.parameters.get("show", True),
            save=self.parameters.get("save", True),
            show_boxes=self.parameters.get("show_boxes", True),
            show_labels=self.parameters.get("show_labels", True),
            conf=self.parameters.get("conf", 0.5),
            iou=self.parameters.get("iou", 0.4),
            persist=self.parameters.get("persist", True)
        )
        return results


class Main:
    def __init__(self, device: str = "cuda"):
        self.device = device

    def train(self, parameters, model_path, data_yaml):
        self.trainer = YOLOv8Trainer(parameters, model_path)
        self.trainer.show_model_info()
        results = self.trainer.train(data_yaml)
        return results

    def validate(self, parameters, model_path, data_yaml):
        self.validator = YOLOv8Validator(parameters, model_path)
        metrics = self.validator.validate(data_yaml)
        print("mAP@0.5:", metrics.box.map50)
        print("mAP@0.5:0.95:", metrics.box.map)
        return metrics

    def infer(self, parameters, model_path, video_path=None):
        self.tracker = YOLOv8Tracker(parameters, model_path)
        results = self.tracker.track_video(video_path)
        return results
