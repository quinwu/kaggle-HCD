import os
import torch


def FreezeParameter(model):
    for parmas in model.parameters():
        parmas.requires_grad = False
    return model


def UnFreezeParameter(model):
    for parmas in model.parameters():
        parmas.requires_grad = True
    return model

def PrintParmeterStatus(model):
    for parmas in model.parameters():
        print (parmas.requires_grad)