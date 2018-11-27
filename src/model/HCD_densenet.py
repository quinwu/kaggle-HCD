import torch
from torchvision import models

def Densenet121(num_classes):
    model_ft = models.densenet121(pretrained=True)
    num_features = model_ft.classifier.in_features
    model_ft.classifier = torch.nn.Linear(num_features, num_classes)
    return model_ft