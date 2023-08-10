import cv2
import torch
from super_gradients.training import models
from super_gradients.common.object_names import  Models
img = cv2.imread("images/image.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
model = models.get('yolo_nas_s', num_classes= 7, checkpoint_path='weights/ckpt_best.pth')#this is just passing the pretrained weights on this yolonas small model

#model.predict_webcam() # to check on the live webcam for the pretective gear
#outputs = model.predict(img) # this is to check the weight on the image defined above 
#outputs.show()
models.convert_to_onnx(model = model, input_shape = (3,640,640), out_path = "custom.onnx") # this is to convert into the onnx format

# onnx format is just a format for all ml models in which after onnx we can convert into any model after that: this is the link to the sample
# https://media.licdn.com/dms/image/D4E12AQGqGhfNviyuWg/article-cover_image-shrink_720_1280/0/1665075477046?e=2147483647&v=beta&t=O9pIQF6bvuzxKu-16EuoWmJXHkZxTIDDpvOa2_5ba3M