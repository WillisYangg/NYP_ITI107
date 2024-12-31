import torch
from ultralytics import YOLO

# check if mps is on the macbook and set the device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device is available. Training will use the MPS backend.")
else:
    device = torch.device("cpu")
    print("MPS device not found. Falling back to CPU.")

model = YOLO("yolov8n.pt")
results = model.train(data='config.yaml', epochs=1000, batch=32, device = device)