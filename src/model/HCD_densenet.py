import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


def Densenet121(num_classes):
    model_ft = models.densenet121(pretrained=True)
    num_features = model_ft.classifier.in_features
    model_ft.classifier = torch.nn.Linear(num_features, num_classes)
    return model_ft

# def Densenet121(num_classes):
#     model_ft = DenseNet(models.densenet121(pretrained=True))
#     num_features = model_ft.classifier.in_features
#     model_ft.classifier = torch.nn.Linear(num_features, num_classes)
#     return model_ft
#
#
# class DenseNet(nn.Module):
#     def __init__(self,model):
#         super(DenseNet, self).__init__()
#         self.features = model.features
#         self.classifier = model.classifier
#     def forward(self, x):
#         features = self.features(x)
#         out = F.relu(features, inplace=True)
#         out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
#         out = self.classifier(out)
#         return out