import os, io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T

class Model():
    def __init__(self, model_path):
        self.model_path = model_path
        
        model = torch.load(model_path)
        model.load_state_dict(torch.load(model_path))
        
        
resnet18_path = 'brain-mri-analysis-to-detect-lesions/Models/savedModels/Resnet18Model.pth'
resnet50_path = 'brain-mri-analysis-to-detect-lesions/Models/savedModels/Resnet50Model.pth'
resnet101_path = 'brain-mri-analysis-to-detect-lesions/Models/savedModels/Resnet101Model.pth'
resnet152_path = 'brain-mri-analysis-to-detect-lesions/Models/savedModels/Resnet152Model.pth'
vgg16_path = 'brain-mri-analysis-to-detect-lesions/Models/savedModels/VGG16Model.pth'
vgg19_path = 'brain-mri-analysis-to-detect-lesions/Models/savedModels/VGG19Model.pth'

print('Done')