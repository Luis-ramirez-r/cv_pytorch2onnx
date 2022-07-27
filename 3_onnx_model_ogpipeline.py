from PIL import Image
import imageio
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
import os
import time


def get_image(path):
    '''
        Using path to image, return the RGB load image
    '''
    img = imageio.imread(path, pilmode='RGB')
    return img

# Pre-processing function for ImageNet models using numpy
def preprocess(img):
    '''
    Preprocessing required on the images for inference with mxnet gluon
    The function takes loaded image and returns processed tensor
    '''
    img = np.array(Image.fromarray(img).resize((224, 224))).astype(np.float32)
    img[:, :, 0] -= 123.68
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 103.939
    img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img

def preprocess(img):
    '''
    Preprocessing required on the images for inference with mxnet gluon
    The function takes loaded image and returns processed tensor
    '''
    img = np.array(Image.fromarray(img).resize((256, 256))).astype(np.float32)
    #center crop
    rm_pad = (256-224)//2 
    img = img[rm_pad:-rm_pad,rm_pad:-rm_pad]
    #normalize to 0-1
    img /= 255.
    #normalize by mean + std
    img = (img - np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225])
    # img[:,:,[0,1,2]] = img[:,:,[2,1,0]] #don't think this is needed?
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img


def predict(path):
    img_batch = preprocess(get_image(path))

    #preprocess = transforms.Compose([
    #    transforms.Resize(256),
    #    transforms.CenterCrop(224),
    #    transforms.ToTensor(),
    #    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #])
    #input_image = Image.open(path )
    #input_tensor = preprocess(input_image)
    #img = np.expand_dims(input_tensor.numpy(), axis=0)
    tik = time.time()
    
    outputs = ort_session.run(
        None,
        {"input": img_batch.astype(np.float32)},
    )

    _ = np.argsort(-outputs[0].flatten())
    results = {}
    for i in _[0:1]:
        results[labels[i]]=float(outputs[0][0][i])
    tok = time.time()
    tt = tok - tik
    
    return results,tt 

ort_session = ort.InferenceSession("./models/googlenet.onnx",providers=['CPUExecutionProvider'])


with open('imagenet_classes.txt', 'r') as f:
    labels = [l.rstrip() for l in f]



times = []
    
for i in range(10):
  tik = time.time()

  image_path = "./images/dog.jpg"
  res,tt=   predict(image_path)

  tok = time.time()

  #print (tok-tik)

  times.append(tt)

for k in res.keys():
  print (k)
