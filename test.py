import json
from prediction import Main

# Load parameters
with open('/content/train_parameters.json', 'r') as file:
    train_parameters = json.load(file)

with open('/content/validation_parameters.json', 'r') as file:
    val_parameters = json.load(file)

with open('/content/infer_parameterts.json', 'r') as file:
    infer_parameters = json.load(file)

# Instantiate the main pipeline
pipeline = Main(device="cuda")

# Train
# pipeline.train(
#     parameters=train_parameters,
#     model_path="yolo11n.pt",
#     data_yaml=r"/content/CRICKET-1/data.yaml"
# )

# Validate
# pipeline.validate(
#     parameters=val_parameters,
#     model_path=r'/content/runs/detect/train/weights/best.pt',
#     data_yaml=r"/content/CRICKET-1/data.yaml"
# )

# Inference
pipeline.infer(
    parameters=infer_parameters,
    model_path="/content/runs/detect/train/weights/best.pt",
    video_path="/content/test.mp4"
)
