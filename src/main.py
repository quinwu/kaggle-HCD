from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.transform.transform import data_transforms
from src.data.dataset import HCDDataset
from src.model.HCD_resnet import Resnet50
from src.train import train_model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root_path = '/home/kwu/data/kaggle/HCD'
train_csv_path = os.path.join(root_path, 'train_labels.csv')
train_data_path = os.path.join(root_path, 'train')
test_data_path = os.path.join(root_path, 'test')

def load_data():
    df = pd.read_csv(train_csv_path)

    data_df = {}
    data_df['train'], data_df['val'] = train_test_split(
        df,
        test_size=0.1,
        random_state=42,
        shuffle=True,
    )

    datasets = {
        x: HCDDataset(
            root=root_path,
            in_df=data_df[x],
            transform=data_transforms[x],
        )
        for x in ['train', 'val']
    }

    dataloaders = {
        x: DataLoader(
            dataset=datasets[x],
            batch_size=128,
            shuffle=True,
            num_workers=4
        )
        for x in ['train', 'val']
    }

    datasets_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

    return dataloaders, datasets_sizes


def pipeline():
    dataloaders, datasets_sizes = load_data()

    model_ft = Resnet50(2)

    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(
        filter(lambda p: p.requires_grad, model_ft.parameters()),
        lr=0.001,
        momentum=0.9
    )

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(
        model=model_ft,
        device=device,
        dataloaders=dataloaders,
        datasets_sizes=datasets_sizes,
        criterion=criterion,
        optimizer=optimizer_ft,
        scheduler=exp_lr_scheduler,
        num_epoches=10
    )

    return model_ft


if __name__ == '__main__':

    print ('device = {}'.format(device))

    pipeline()