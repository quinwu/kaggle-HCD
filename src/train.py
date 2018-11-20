import copy
import time
import torch
import numpy as np
from tqdm import tqdm

def train_model(model, device, dataloaders, datasets_sizes, criterion, optimizer, scheduler, num_epoches=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epoches):
        epoch_acc = {}

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_corrects = 0
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

                tq.update(batch_size)
                # running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                mean_loss = np.mean(losses[-50:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))


            epoch_acc[phase] = (running_corrects.double() / datasets_sizes[phase]).item()
            tq.close()

            if phase == 'val' and epoch_acc['val'] > best_acc:
                best_acc = epoch_acc['val']
                best_model_wts = copy.deepcopy(model.state_dict())

        print ('train  Acc: {:.4f}, val Acc: {:.4f}'.format(epoch_acc['train'],epoch_acc['val']))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60
    ))
    print('Best val ACC :{:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)

    return model
