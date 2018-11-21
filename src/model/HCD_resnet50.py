import torch
from torchvision import models

def Resnet18(num_classes):
    model_ft = models.resnet18(pretrained=True)
    num_features = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_features, num_classes)
    model_ft.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
    return model_ft

def Resnet34(num_classes):
    model_ft = models.resnet34(pretrained=True)
    num_features = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_features, num_classes)
    model_ft.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
    return model_ft

def Resnet50(num_classes):
    model_ft = models.resnet50(pretrained=True)
    num_features = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_features, num_classes)
    model_ft.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
    return model_ft

def Resnet101(num_classes):
    model_ft = models.resnet101(pretrained=True)
    num_features = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_features, num_classes)
    model_ft.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
    return model_ft

def Resnet152(num_classes):
    model_ft = models.resnet152(pretrained=True)
    num_features = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_features, num_classes)
    model_ft.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
    return model_ft