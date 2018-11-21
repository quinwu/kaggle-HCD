import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from src.data.dataset import HCDDataset
from src.transform.transform import data_transforms

from tqdm import tqdm

def load_test_data(root):
    csv_path = os.path.join(root,'sample_submission.csv')
    df = pd.read_csv(csv_path)

    print ('len(df) = {}'.format(len(df)))

    dataset = HCDDataset(
        root=root,
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

def predict(model, device, dataloader, path):

    model = model.to(device)
    model.eval()

    ids = []
    preds = []

    batch_size = dataloader.batch_size
    tq = tqdm(total=len(dataloader)*batch_size)
    tq.set_description('predict')

    for index,data in enumerate(dataloader):

        inputs, ids_batch = data
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds_batch = torch.max(outputs, 1)

        preds_batch = preds_batch.tolist()
        ids_batch = list(ids_batch)

        ids += ids_batch
        preds += preds_batch

        tq.update(batch_size)

    tq.close()

    df = pd.DataFrame({'id':ids, 'label':preds})
    df.to_csv(path,index=False)
    print (df.head())


