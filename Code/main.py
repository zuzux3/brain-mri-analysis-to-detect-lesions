import cv2
import numpy as np
from PIL import Image
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import torchvision.models as models

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

#models paths
resnet18_path = 'brain-mri-analysis-to-detect-lesions/Models/savedModels/Resnet18Model.pth'
resnet50_path = 'brain-mri-analysis-to-detect-lesions/Models/savedModels/Resnet50Model.pth'
resnet101_path = 'brain-mri-analysis-to-detect-lesions/Models/savedModels/Resnet101Model.pth'
resnet152_path = 'brain-mri-analysis-to-detect-lesions/Models/savedModels/Resnet152Model.pth'
vgg16_path = 'brain-mri-analysis-to-detect-lesions/Models/savedModels/VGG16Model.pth'
vgg19_path = 'brain-mri-analysis-to-detect-lesions/Models/savedModels/VGG19Model.pth'
yolov8_path = 'brain-mri-analysis-to-detect-lesions/Models/savedModels/yolov8.pt'

#classes
classes = {
    0: 'Glioma Tumor',
    1: 'Meningioma Tumor',
    2: 'No Tumor',
    3: 'Pituitary Tumor'
}

num_classes = len(classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# Grad-CAM
# =========================

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

# =========================
# Preprocessing
# =========================

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

# =========================
# Model Loading
# =========================


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

# =========================
# Grad-CAM overlay (TAB-1)
# =========================

def overlay_cam_on_image(img, cam, alpha=0.4):
    h, w, _ = img.shape
    cam_resized = cv2.resize(cam, (w, h))
    
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = np.uint8(alpha * heatmap + (1 - alpha) * img)
    
    return overlay    
    
def classify_gradcam(gradcam_obj, model_name, img, class_id=2, device='cpu'):
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

def classify_img_gradcam(img):
    results_flat = []
    
    for gc, name in zip(gradcams, models_names):
        cam_img, label = classify_gradcam(gc, name, img)
        results_flat.extend([cam_img, label])
        
    return results_flat

# =========================
# OpenCV Bounding Box (TAB-2)
# =========================

def cam_to_bbox_img(img, cam, threshold=0.4):
    h, w = img.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    _, cam_bin = cv2.threshold(
        (cam_resized * 255).astype(np.uint8),
        int(threshold * 255),
        255,
        cv2.THRESH_BINARY
    )
    
    contours, _ = cv2.findContours(cam_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_img = img.copy()
    
    if contours is None or len(contours) == 0:
        return output_img    
    
    c = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(c) < 100:
        return output_img
    
    x, y, bw, bh = cv2.boundingRect(c)
    cv2.rectangle(output_img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
    
    return output_img

def classify_opencv_bbox(gradcam_obj, model_name, img, class_id=2, device='cpu'):
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
        out_img = cam_to_bbox_img(img, cam_np)
        
    return out_img, label_text

def classify_img_opencv(img):
    results_flat = []
    
    for gc, name in zip(gradcams, models_names):
        bbox_img, label = classify_opencv_bbox(gc, name, img)
        
        results_flat.extend([bbox_img, label])
        
    return results_flat

# =========================
# YOLOv8 + classification (TAB-3)
# =========================

if YOLO_AVAILABLE:
    try:
        yolo_model = YOLO(yolov8_path)
    except Exception as e:
        print(f"Error loading YOLOv8 model: {e}")
        yolo_model = None
        
else:
    yolo_model = None
    print("Ultralytics YOLO package not available. YOLOv8 functionality will be disabled.")
    
def ensamble_classify_patch(patch_np):
    patch_pil = Image.fromarray(patch_np.astype('uint8'))
    x = preprocess_image(patch_pil).to(device)
    
    model_results = []
    all_probs = []
    
    with torch.no_grad():
        for m, name in zip(models_list, models_names):
            logits = m(x)
            probs = F.softmax(logits, dim=1)
            probs_np = probs.detach().cpu().numpy()[0]
            
            all_probs.append(probs_np)
            
            pred_idx = int(probs_np.argmax())
            pred_prob = float(probs_np[pred_idx])
            model_results.append((name, pred_idx, pred_prob))
            
    all_probs = np.stack(all_probs, axis=0)
    mean_probs = all_probs.mean(axis=0)
    
    ens_idx = int(mean_probs.argmax())
    ens_prob = float(mean_probs[ens_idx])
    
    return model_results, ens_idx, ens_prob

def yolo_detect(img):
    if yolo_model is None:
        return img, 'YOLOv8 model not available.'
    
    img_rgb = img.copy()
    output = img_rgb.copy()
    
    results = yolo_model(img_rgb, verbose=False)
    
    if len(results) == 0:
        return output, 'No detections.'
    
    result = results[0]
    
    if result.boxes is None or len(result.boxes) == 0:
        return output, 'No bounding boxes detected.'
    
    names = result.names
    msgs = []
    
    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = box.conf[0].cpu().numpy()
        cls_id = int(box.cls[0].cpu().numpy())
        cls_name = names.get(cls_id, f'class{cls_id}')
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1] - 1, x2)
        y2 = min(img.shape[0] - 1, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            output,
            f'{cls_name} {conf:.2f}',
            (x1, max(0, y1-5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )
        
        msgs.append(f'Detection {i+1}: {cls_name} (conf: {conf:.2f})')
        
        if not msgs:
            msg = 'No valid detections.'
        else:
            msg = '\n'.join(msgs)
            
        return output, msg



# =========================
# Gradio UI
# =========================

with gr.Blocks() as ui:
    gr.Markdown('Brain MRI Analysis to detect lesions')
    
    with gr.Tabs():
        with gr.Tab('Grad-CAM'):
            gr.Markdown('Lesion detection using Grad-Cam with pretrained models')
            
            in_img = gr.Image(type='numpy', label='Input Brain MRI Image')
            
            outs = []
            
            for i in range(len(models_names)):
                with gr.Row():
                    out_img = gr.Image(type='numpy', label=f'{models_names[i]} - CAM/Plain')
                    out_label = gr.Textbox(label=f'{models_names[i]} - Prediction')
                    
                outs.extend([out_img, out_label])
                
            run_btn = gr.Button('Analyze Image')
            run_btn.click(
                fn=classify_img_gradcam,
                inputs=in_img,
                outputs=outs
            )
            
        with gr.Tab('OpenCV Bounding Box'):
            gr.Markdown('Lesion detection using OpenCV bounding box')
            
            in_img2 = gr.Image(type='numpy', label='Input Brain MRI Image')
            
            outs2 = []
            
            for i in range(len(models_names)):
                with gr.Row():
                    out_img2 = gr.Image(type='numpy', label=f'{models_names[i]} - BBox/Plain')
                    out_label2 = gr.Textbox(label=f'{models_names[i]} - Prediction')
                    
                outs2.extend([out_img2, out_label2])
                
            run_btn2 = gr.Button('Analyze Image')
            run_btn2.click(
                fn=classify_img_opencv,
                inputs=in_img2,
                outputs=outs2
            )
            
        with gr.Tab('YOLOv8 + Classification'):
            gr.Markdown('Lesion detection using YOLOv8 and classification ensemble')
            
            in_img3 = gr.Image(type='numpy', label='Input Brain MRI Image')
            out_img3 = gr.Image(type='numpy', label='Output Image with Detections')
            out_text3 = gr.Textbox(label='Detections Summary')
            out_lb3 = gr.Textbox(label=f'{models_names[i]} - Prediction')
            
            run_btn3 = gr.Button('Analyze Image')
            run_btn3.click(
                fn=yolo_detect,
                inputs=in_img3,
                outputs=[out_img3, out_text3, out_lb3]
            )
'''with gr.Blocks() as ui: 
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
        fn=classify_img_gradcam,
        inputs=in_img,
        outputs=outs
    )'''

ui.launch(share=True)