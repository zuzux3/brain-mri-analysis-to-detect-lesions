import os, io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import torchvision.models as models

#models paths
resnet18_path = 'brain-mri-analysis-to-detect-lesions/Models/savedModels/Resnet18Model.pth'
resnet50_path = 'brain-mri-analysis-to-detect-lesions/Models/savedModels/Resnet50Model.pth'
resnet101_path = 'brain-mri-analysis-to-detect-lesions/Models/savedModels/Resnet101Model.pth'
resnet152_path = 'brain-mri-analysis-to-detect-lesions/Models/savedModels/Resnet152Model.pth'
vgg16_path = 'brain-mri-analysis-to-detect-lesions/Models/savedModels/VGG16Model.pth'
vgg19_path = 'brain-mri-analysis-to-detect-lesions/Models/savedModels/VGG19Model.pth'

#classes
classes = {
    0: 'Glioma Tumor',
    1: 'Meningioma Tumor',
    2: 'No Tumor',
    3: 'Pituitary Tumor'
}

num_classes = len(classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        def forward_hook(module, inp, out):
            self.activations = out.detach()
            
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
            
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
        
    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        
        logits = self.model(x)
        
        if class_idx is None:
            class_idx = logits.argmax(1).item()
        
        score = logits[0, class_idx]
        score.backward()
        
        grads = self.gradients[0]
        acts = self.activations[0]
        
        weights = grads.mean(dim=(1, 2))
        cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=acts.device)
        
        for k, w in enumerate(weights):
            cam += w * acts[k]
            
        cam = F.relu(cam)
        cam = cam.detach().cpu()
        
        if cam.max() > 0:
            cam = cam / cam.max()
            
        cam_np = cam.numpy()
        
        return logits, cam_np

def preprocess_image(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(img).unsqueeze(0)
    return tensor

def get_last_conv_(model, isResnet: bool = True):
    if isResnet:
        return model.layer4[-1]
    else:
        last_conv = None
        
        for m in model.features:
            if isinstance(m, nn.Conv2d):
                last_conv = m
                
        return last_conv
    
def load_models():
    resnet18 = models.resnet18(weights=None)
    resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
    resnet18.load_state_dict(torch.load(resnet18_path, map_location=device))
    resnet18.to(device)
    
    resnet50 = models.resnet50(weights=None)
    resnet50.fc = nn.Linear(resnet50.fc.in_features, num_classes)
    resnet50.load_state_dict(torch.load(resnet50_path, map_location=device))
    resnet50.to(device)
    
    resnet101 = models.resnet101(weights=None)
    resnet101.fc = nn.Linear(resnet101.fc.in_features, num_classes)
    resnet101.load_state_dict(torch.load(resnet101_path, map_location=device))
    resnet101.to(device)
    
    resnet152 = models.resnet152(weights=None)
    resnet152.fc = nn.Linear(resnet152.fc.in_features, num_classes)
    resnet152.load_state_dict(torch.load(resnet152_path, map_location=device))
    resnet152.to(device)
    
    vgg16 = models.vgg16_bn(weights=None)
    vgg16.classifier[6] = nn.Linear(vgg16.classifier[6].in_features, num_classes)
    vgg16.load_state_dict(torch.load(vgg16_path, map_location=device))
    vgg16.to(device)
    
    vgg19 = models.vgg19_bn(weights=None)
    vgg19.classifier[6] = nn.Linear(vgg19.classifier[6].in_features, num_classes)
    vgg19.load_state_dict(torch.load(vgg19_path, map_location=device))
    vgg19.to(device)
    
    models_list = [resnet18, resnet50, resnet101, resnet152, vgg16, vgg19]
    models_names = ['ResNet18', 'ResNet50', 'ResNet101', 'ResNet152', 'VGG16', 'VGG19']
    
    return models_list, models_names

models_list, models_names = load_models()

target_layers = [
    get_last_conv_(models_list[0], isResnet=True),
    get_last_conv_(models_list[1], isResnet=True),
    get_last_conv_(models_list[2], isResnet=True),
    get_last_conv_(models_list[3], isResnet=True),
    get_last_conv_(models_list[4], isResnet=False),
    get_last_conv_(models_list[5], isResnet=False)
]

gradcams = [
    GradCAM(m, layer) for m, layer in zip(models_list, target_layers)
]
    
def overlay_cam_on_image(img, cam, alpha=0.4):
    h, w, _ = img.shape
    cam_resized = cv2.resize(cam, (w, h))
    
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = np.uint8(alpha * heatmap + (1 - alpha) * img)
    
    return overlay    
    
def classify(gradcam_obj, model_name, img, class_id=2, device='cpu'):
    pil_img = Image.fromarray(img.astype('uint8'))
    x = preprocess_image(pil_img).to(device)
    
    logits, cam_np = gradcam_obj(x)
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    
    pred_idx = probs.argmax(1).item()
    pred_prob = probs[0, pred_idx].item()
    
    label_text = f'Predicted class: {classes[pred_idx]}'
    
    if pred_idx == class_id:
        out_img = img
    
    else:
        out_img = overlay_cam_on_image(img, cam_np)
        
    return out_img, label_text

def classify_img(img):
    results_flat = []
    
    for gc, name in zip(gradcams, models_names):
        cam_img, label = classify(gc, name, img)
        results_flat.extend([cam_img, label])
        
    return results_flat

with gr.Blocks() as ui: 
    gr.Markdown('Brain MRI Analysis to detect Lesions using Grad-CAM')
    in_img = gr.Image(type='numpy', label='Input Brain MRI Image')
    
    outs = []
    for i in range(len(models_names)):
        with gr.Row():
            out_img = gr.Image(type='numpy', label=f'{models_names[i]} - CAM/Plain')
            out_label = gr.Textbox(label=f'{models_names[i]} - Prediction')
            
        outs.extend([out_img, out_label])
        
    run_btn = gr.Button('Analyze Image')
    
    run_btn.click(
        fn=classify_img,
        inputs=in_img,
        outputs=outs
    )

ui.launch(share=True)