import os
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from src.data.dataset import *

root_path = '/home/kwu/data/kaggle/HCD'
train_data_path = '/home/kwu/data/kaggle/HCD/train'
test_data_path = '/home/kwu/data/kaggle/HCD/test'

BATCH_SIZE = 256

def load_data_cv(df,data_transform):

    datasets = {
        x:HCDDataset(
            root=train_data_path,
            in_df=df[x],
            transform=data_transform[x],
        )
        for x in ['train', 'val']
    }

    dataloaders = {
        x:DataLoader(
            dataset=datasets[x],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4
        )
        for x in ['train', 'val']
    }

    return dataloaders

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
            root=train_data_path,
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


def load_test_data(root,data_transforms):
    csv_path = os.path.join(root,'sample_submission.csv')
    df = pd.read_csv(csv_path)

    dataset = HCDDataset(
        root=test_data_path,
        in_df=df,
        transform=data_transforms['test'],
        mode='test'
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=128,
        num_workers=4
    )

    return dataloader


def tta_load_test_data(root, data_transforms):
    csv_path = os.path.join(root,'sample_submission.csv')
    df = pd.read_csv(csv_path)

    dataset = HCDDataset(
        root=test_data_path,
        in_df=df,
        transform=data_transforms['test'],
        mode='test'
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=128,
        num_workers=4,
    )

    return dataloader


def load_stack_data(csv_path,data_transform):

    # train_df, stack_df = train_test_split(df,
    #                                       test_size=0.35,
    #                                       random_state=42,
    #                                       shuffle=True)
    #
    # train_df.to_csv('/home/kwu/Project/kaggle/HCD/stack/train.csv', index=False)
    # stack_df.to_csv('/home/kwu/Project/kaggle/HCD/stack/stack.csv', index=False)


    train_csv_path = os.path.join(csv_path, 'train.csv')
    stack_csv_path = os.path.join(csv_path, 'stack.csv')

    train_df = pd.read_csv(train_csv_path)
    stack_df = pd.read_csv(stack_csv_path)

    stackdataset = HCDDataset(
        root=train_data_path,
        in_df=stack_df,
        transform=data_transform['test'],
        mode='test'
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
            root=train_data_path,
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


def load_ensemble_stack_data(stack_csv_path,ensemble_csv_path):
    # stack_csv_path = '/home/kwu/Project/kaggle/HCD/data/stack.csv'
    # ensemble_csv_path = '/home/kwu/Project/kaggle/HCD/stack'

    df = pd.read_csv(stack_csv_path)

    data_df = {}

    data_df['train'], data_df['val'] = train_test_split(
        df,
        test_size=0.15,
        random_state=42,
        shuffle=True
    )

    datasets = {
        x : StackDataset(
            stack_root=ensemble_csv_path,
            in_df=data_df[x],
        )
        for x in ('train', 'val')
    }

    dataloaders = {
        x : DataLoader(
            dataset=datasets[x],
            batch_size=4,
            shuffle=True,
            num_workers=4
        )
        for x in ('train', 'val')
    }

    return dataloaders

def load_ensemble_stack_test_data(ensemble_csv_path):
    csv_path = '/home/kwu/data/kaggle/HCD/sample_submission.csv'
    # ensemble_csv_path = '/home/kwu/Project/kaggle/HCD/stack'

    df = pd.read_csv(csv_path)

    dataset = StackDataset(
        stack_root=ensemble_csv_path,
        in_df=df,
        mode='test'
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=256,
        num_workers=4
    )

    return dataloader

def split_cnn_stack(scale):
    df = pd.read_csv('/home/kwu/data/kaggle/HCD/train_labels.csv')
    print (df.shape)
    train_df, stack_df = train_test_split(df,
                                          test_size=scale,
                                          random_state=42,
                                          shuffle=True)
    print (train_df.shape)
    print (stack_df.shape)

    train_csv_path = os.path.join(os.path.join('/home/kwu/Project/kaggle/HCD/data/', str(int(100*scale))), 'train.csv')
    stack_csv_path = os.path.join(os.path.join('/home/kwu/Project/kaggle/HCD/data/', str(int(scale*100))), 'stack.csv')

    train_df.to_csv(train_csv_path, index=False)
    stack_df.to_csv(stack_csv_path, index=False)


# #
# if __name__ == '__main__':
#     split_cnn_stack(0.25)