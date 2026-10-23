import json, torch, torch.nn as nn, gradio as gr
from torchvision import models, transforms

class_names = json.load(open("class_names.json"))
model = models.resnet50()
model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, len(class_names)))
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict(img):
    with torch.no_grad():
        probs = torch.softmax(model(tf(img.convert("RGB")).unsqueeze(0)), 1)[0]
    return {c: float(p) for c, p in zip(class_names, probs)}

gr.Interface(predict, gr.Image(type="pil"), gr.Label(num_top_classes=3)).launch()
