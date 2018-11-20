import torch
from torchvision import models


def Resnet50(num_classes):

    model_ft = models.resnet50(pretrained=True)
    num_features = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_features,num_classes)
    model_ft.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))

    return model_ft