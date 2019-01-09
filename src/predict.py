import os
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from src.transform.transform import data_transforms2,data_transforms1
from torchvision import transforms
from .transform.tta import ClassPredictor


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


def tta_predict(model, device, dataloader, path):
    model = model.to(device)
    model.eval()

    tta_predictor = ClassPredictor(model=model, device=device)

    ids = []
    preds = []

    batch_size = dataloader.batch_size
    tq = tqdm(total=len(dataloader) * batch_size)
    tq.set_description('predict')

    for index, data in enumerate(dataloader):
        inputs, ids_batch = data

        preds_batch = tta_predictor(inputs)
        ids_batch = list(ids_batch)

        ids += ids_batch
        preds += preds_batch
        tq.update(batch_size)

    tq.close()
    df = pd.DataFrame({'id':ids, 'label':preds})
    df.to_csv(path,index=False)
    print (df.head())
