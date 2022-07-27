import time
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import onnx


PATH = 'models/googlenet.pt'
model = torch.load(PATH)

output_model_file = 'models/googlenet.onnx'

filename = './images/dog.jpg'



input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model


torch.onnx.export(model,               # model being run
                  input_batch,                         # model input (or a tuple for multiple inputs)
                  output_model_file,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'] # the model's output names
                  )



onnx_model = onnx.load(output_model_file)
onnx.checker.check_model(onnx_model)

print ('The model has been saved at: models/googlenet.onnx')

