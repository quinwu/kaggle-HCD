import torch
import torch.nn as nn
from torchvision import models
from .utils import FreezeParameter,UnFreezeParameter,PrintParmeterStatus

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Classifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features

        self.classifier = nn.Sequential(
            Flatten(),
            nn.BatchNorm1d(num_features=self.num_features*2),
            nn.Dropout(p=0.8),
            nn.Linear(in_features=self.num_features*2, out_features=512, bias=True),
            nn.ReLU(inplace=True),

            nn.BatchNorm1d(num_features=512),
            nn.Dropout(p=0.8),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.ReLU(inplace=True),

            nn.BatchNorm1d(num_features=256),
            nn.Dropout(p=0.8),
            nn.Linear(in_features=256, out_features=num_classes, bias=True)
        )
    def forward(self, x):
        return self.classifier(x)

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
    model_ft = FreezeParameter(model_ft)
    num_features = model_ft.fc.in_features
    model_ft.avgpool = AdaptiveConcatPool2d(1)
    classifier = Classifier(num_features, num_classes)
    model_ft.fc = classifier
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

