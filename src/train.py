import copy
import time
import torch
import numpy as np
from tqdm import tqdm

import torch.nn.functional as F


def semi_supervised_train():
    pass


def train_model(model, device, dataloaders,
                criterion, optimizer, scheduler, num_epoches=10):
    since = time.time()

    # best_model_wts = copy.deepcopy(model.state_dict())
    # best_measure = {}
    # best_measure['acc'] = 0.0

    for epoch in range(num_epoches):
        epoch_acc = {}
        # epoch_sensitivity = {}
        # epoch_specificity = {}

        for phase in ['train', 'val']:
            evalutions = {}
            evalutions['TP'] = 0
            evalutions['TN'] = 0
            evalutions['FP'] = 0
            evalutions['FN'] = 0

            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            losses = []
            batch_size = dataloaders[phase].batch_size
            tq = tqdm(total=len(dataloaders[phase]) * batch_size)
            tq.set_description('Epoch {phase} {epoch} '.format(phase=phase,epoch=epoch))

            for index, data in enumerate(dataloaders[phase]):

                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                losses.append(loss.item())

                evalutions['TP'] += torch.sum((preds == 1) & (labels.data == 1)).cpu().item()
                evalutions['TN'] += torch.sum((preds == 0) & (labels.data == 0)).cpu().item()
                evalutions['FP'] += torch.sum((preds == 1) & (labels.data == 0)).cpu().item()
                evalutions['FN'] += torch.sum((preds == 0) & (labels.data == 1)).cpu().item()

                tq.update(batch_size)
                mean_loss = np.mean(losses[-50:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))

            epoch_acc[phase] = (evalutions['TP'] + evalutions['TN']) / (evalutions['TP'] + evalutions['TN'] + evalutions['FP'] + evalutions['FN'])
            # epoch_sensitivity[phase] = (evalutions['TP'] ) / (evalutions['TP'] + evalutions['FN'])
            # epoch_specificity[phase] = (evalutions['TN']) / (evalutions['TN'] + evalutions['FP'])
            tq.close()

            # if phase == 'val' and epoch_acc['val'] > best_measure['acc']:
            #     best_measure['acc'] = epoch_acc['val']
            #     best_measure['sensitivity'] = epoch_sensitivity['val']
            #     best_measure['specificity'] = epoch_specificity['val']
                # best_model_wts = copy.deepcopy(model.state_dict())

        print('train ACC: {:.4f}, val ACC： {:.4f}'.format(epoch_acc['train'], epoch_acc['val']))
        # print('train sensitivity: {:.4f}, val sensitivity: {:.4f}'.format(epoch_sensitivity['train'], epoch_sensitivity['val']))
        # print('train specificity: {:.4f}, val specificity: {:.4f}'.format(epoch_specificity['train'], epoch_specificity['val']))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60
    ))

    # print('Best val ACC :{:4f}, sensitivity :{:.4f}, specificity :{:.4f}'.format(best_measure['acc'], best_measure['sensitivity'], best_measure['specificity']))

    # model.load_state_dict(best_model_wts)

    return model



