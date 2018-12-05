import os
import PIL
import numpy as np
from torchvision import transforms
from .utils import *
import torch.nn.functional as F


tta_aug = [
    Hflip(),
    Vflip(),
    Rotate(90),
    Rotate(180),
]

class ClassPredictor():

    def __init__(self, model,device, augs = tta_aug):
        self.model = model
        self.augs = augs
        self.device = device
        self.preds = []

    def predict(self,inputs):

        imgs = TensorToPILs(inputs)

        for aug in self.augs:
            self.preds.append(self._predict_single(imgs,aug))

        print (self.preds)

    def _predict_single(self, imgs, aug):
        aug_imgs = aug(imgs)
        inputs = PILsToTensor(aug_imgs)
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)

        outputs = F.softmax(outputs,1)
        preds_batch = outputs[:,1].tolist()

        print (type(preds_batch))
        print (len(preds_batch))

        return preds_batch

    def __call__(self, inputs):
        self.predict(inputs)

# if __name__ == '__main__':
#     model = 'test'
#     test = ClassPredictor(model)

