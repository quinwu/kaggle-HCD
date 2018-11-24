import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.data.dataset import HCDDataset
import torch.nn.functional as F
from src.transform.transform import data_transforms2,data_transforms1

def load_test_data(root):
    csv_path = os.path.join(root,'sample_submission.csv')
    df = pd.read_csv(csv_path)

    dataset = HCDDataset(
        root=root,
        in_df=df,
        transform=data_transforms1['test'],
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

        outputs = F.softmax(outputs,1)
        preds_batch = outputs[:,1].tolist()
        ids_batch = list(ids_batch)

        ids += ids_batch
        preds += preds_batch

        tq.update(batch_size)

    tq.close()
    df = pd.DataFrame({'id':ids, 'label':preds})
    df.to_csv(path,index=False)
    print (df.head())


