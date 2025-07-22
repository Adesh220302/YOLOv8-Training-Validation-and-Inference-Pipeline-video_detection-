
#YOLOn11 Training, Validation, and Inference Pipeline
This project provides a modular and class-based pipeline to train, validate, and run inference/tracking using YOLOv8.

📁 Project Structure
prediction.py: Contains class-based implementations for training, validation, and inference using the YOLOv8 model.

test.py: Script to run training, validation, and inference using external JSON parameter files.

🚀 Features
YOLOv8 model integration via ultralytics package.

Modular design with YOLOv8Trainer, YOLOv8Validator, YOLOv8Tracker, and Main controller classes.

JSON-based configuration for flexible parameter management.

Easy-to-switch between training, validation, and inference modes.

🛠️ Requirements
Python 3.8+

ultralytics

torch

opencv-python (for tracking visualization, if used)

Install dependencies:

bash
Copy
Edit
pip install ultralytics opencv-python
⚙️ Usage
Create or modify your configuration files:

train_parameters.json

validation_parameters.json

infer_parameters.json

Modify and run test.py as needed:

python
Copy
Edit
# Training
pipeline.train(parameters=train_parameters,
               model_path="yolo11n.pt",
               data_yaml="/path/to/data.yaml")

# Validation
pipeline.validate(parameters=val_parameters,
                  model_path="/path/to/best.pt",
                  data_yaml="/path/to/data.yaml")

# Inference
pipeline.infer(parameters=infer_parameters,
               model_path="/path/to/best.pt",
               video_path="/path/to/video.mp4")
📌 Notes
Make sure your paths are correct, especially for the model weights and dataset.

Adjust the device key for GPU (0) or CPU ("cpu").

You can easily switch between train, validate, and infer modes by commenting/uncommenting lines in test.py.

📸 Output
Tracked results and inference visuals will be saved if save=True is set in the parameters.

