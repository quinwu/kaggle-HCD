import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class HCDDataset(Dataset):
    def __init__(self,root,in_df,transform=None,mode='train'):
        self.root = root 
      
        self.image_ids = list(in_df['id'])
        self.labels = list(in_df['label'])
        
        self.transform = transform
        self.mode = mode

    def __getitem__(self,item):
        image_id = self.image_ids[item]
        if self.mode == 'train':
            train_path = os.path.join(self.root,'train')
            image_path = os.path.join(train_path,image_id) + '.tif'
        else:
            test_path = os.path.join(self.root,'test')
            image_path = os.path.join(test_path,image_id) + '.tif'

        image = Image.open(image_path)
        label = np.array(self.labels[item])

        if self.transform is not None:
            image = self.transform(image)
        
        if self.mode == 'train':
            return image, torch.from_numpy(label)
        else:
            return image, str(image_id)

    def __len__(self):
        return len(self.image_ids)
