"""
This script is used to export a PyTorch model to the ONNX format for deployment or inference in environments that support ONNX.
It ensures that the model is in evaluation mode and uses a dummy input to trace the model's computation graph.
"""

import torch
from models.resnet import ChessResNet

# hyper params
batch_size = 4096
input_channels = 20
num_blocks = 5
num_classes = 1

model = ChessResNet(input_channels=input_channels, num_blocks=num_blocks, num_classes=num_classes)

state_dict = torch.load('chess_resnet_epoch_20.pt', map_location='cpu')

model.load_state_dict(state_dict)

model.eval()

dummy = torch.randn(1, 20, 8, 8)

torch.onnx.export(
    model, 
    dummy, 
    "chess_evaluator.onnx", 
    dynamo=True
)