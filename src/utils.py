import os
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .data.dataset import *

root_path = '/home/kwu/data/kaggle/HCD'

BATCH_SIZE = 512

def load_data(df,data_transform):

    data_df = {}
    data_df['train'], data_df['val'] = train_test_split(
        df,
        test_size=0.05,
        random_state=42,
        shuffle=True,
    )

    datasets = {
        x : HCDDataset(
            root=root_path,
            in_df=data_df[x],
            transform=data_transform[x],
        )
        for x in ['train', 'val']
    }

    dataloaders = {
        x :DataLoader(
            dataset=datasets[x],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4
        )
        for x in ['train', 'val']
    }

    datasets_sizes = {x : len(datasets[x]) for x in ['train', 'val']}

    return dataloaders, datasets_sizes


def load_stack_data(data_transform):

    # train_df, stack_df = train_test_split(df,
    #                                       test_size=0.35,
    #                                       random_state=42,
    #                                       shuffle=True)
    #
    # train_df.to_csv('/home/kwu/Project/kaggle/HCD/stack/train.csv', index=False)
    # stack_df.to_csv('/home/kwu/Project/kaggle/HCD/stack/stack.csv', index=False)

    train_csv_path = '/home/kwu/Project/kaggle/HCD/stack/train.csv'
    stack_csv_path = '/home/kwu/Project/kaggle/HCD/stack/stack.csv'

    train_df = pd.read_csv(train_csv_path)
    stack_df = pd.read_csv(stack_csv_path)

    stackdataset = HCDDataset(
        root=root_path,
        in_df=stack_df,
        transform=data_transform['train'],
    )

    stackdataloader = DataLoader(
        dataset=stackdataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    data_df = {}
    data_df['train'], data_df['val'] = train_test_split(
        train_df,
        test_size=0.05,
        random_state=42,
        shuffle=True,
    )

    datasets = {
        x : HCDDataset(
            root=root_path,
            in_df=data_df[x],
            transform=data_transform[x],
        )
        for x in ['train', 'val']
    }

    dataloaders = {
        x :DataLoader(
            dataset=datasets[x],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4
        )
        for x in ['train', 'val']
    }

    return dataloaders,stackdataloader