# -*- coding: utf-8 -*-
import torch

def label_encoder(y, labels):
    new_y = y.copy()
    for i, label in enumerate(labels):
        ix = (y == label)
        new_y[ix] = i
    return new_y

def generate_tensors(*args, dtype=torch.float):
    new_args = []
    for arg in args:
        new_args.append(torch.as_tensor(arg, dtype=dtype))
    if len(new_args) == 1:
        return new_args[0]
    else:
        return new_args
