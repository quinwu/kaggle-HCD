import os
import PIL
import numpy as np

def apply(aug,img):
    pass


class BasePredictor():

    def __init__(self,model):
        self.model = model


    def apply_aug(self,img):
        aug_patch = []
        for aug in enumerate(self.augs):
            aug_patch.append(apply(aug,img))
        return aug_patch

    def predict_patches(self,patches):
        pass

    def _predict_single(self,*input):
        raise NotImplementedError

    def predict_images(self,imgs):
        preds = []
        for img in imgs:
            pred = self._predict_single(img)
            preds.append(pred)
        return np.array(preds)

class ClassPredictor(BasePredictor):

    def __init__(self,model):
        self.model = model

    def _predict_single(self,img):
        pass



if __name__ == '__main__':
    model = 'test'
    test = ClassPredictor(model)