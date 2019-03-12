import os
import torch
import pandas as pd
import numpy as np
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
        # if self.mode == 'train':
        #     train_path = os.path.join(self.root,'train')
        #     image_path = os.path.join(train_path,image_id) + '.tif'
        # if self.mode == 'test':
        #     test_path = os.path.join(self.root,'test')
        image_path = os.path.join(self.root,image_id) + '.tif'

        # image = np.asarray(Image.open(image_path))
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


class StackDataset(Dataset):
    def __init__(self, stack_root, in_df, mode='train'):
        self.root = stack_root
        self.mode = mode
        self.df = in_df

        self.ids = list(in_df['id'])
        self.labels = list(in_df['label'])
        self.files = os.listdir(self.root)
        self._merge_csv()

    def __getitem__(self, item):
        id = self.ids[item]
        data = self.data[item]
        label = np.array(self.labels[item])

        if self.mode == 'train':
            return torch.from_numpy(data), torch.from_numpy(label)
        else:
            return torch.from_numpy(data), str(id)

    def __len__(self):
        return len(self.ids)

    def _merge_csv(self):
        merge = []
        for fn in os.listdir(self.root):
            sub = pd.read_csv(os.path.join(self.root, fn)).set_index('id')
            labels = []
            for i in self.ids:
                labels.append(sub.loc[i]['label'])
            merge.append(labels)
        merge = np.asarray(merge)
        self.data = np.transpose(merge)