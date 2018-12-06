import os
import PIL
import numpy as np
from torchvision import transforms
from .utils import *
import torch.nn.functional as F


tta_aug = [
    NoneAug(),
    Hflip(),
    Vflip(),
    Rotate(90),
    Rotate(180),
    Rotate(270),
]

class ClassPredictor():

    def __init__(self, model,device, augs = tta_aug):
        self.model = model
        self.augs = augs
        self.device = device

    def predict(self,inputs):
        self.preds = []
        imgs = TensorToPILs(inputs)

        for aug in self.augs:
            self.preds.append(self._predict_single(imgs,aug))
        self.preds = np.mean(np.array(self.preds),axis=0)

        return self.preds.tolist()

    def _predict_single(self, imgs, aug):
        aug_imgs = aug(imgs)
        inputs = PILsToTensor(aug_imgs)
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)

        outputs = F.softmax(outputs,1)
        preds_batch = outputs[:,1].tolist()

        return preds_batch

    def __call__(self, inputs):
        return self.predict(inputs)


