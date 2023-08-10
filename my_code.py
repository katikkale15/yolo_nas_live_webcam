import cv2
import torch
from super_gradients.training import models
from super_gradients.common.object_names import Models

# note that currently yolox and ppyoloe are supported

model= models.get(Models.YOLOX_M, pretrained_weights='coco')

model=model.to("cuda" if torch.cuda.is_available() else 'cpu')

model.predict_webcam()