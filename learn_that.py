from typing import Any, Dict

import numpy as np
import rtdl
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import zero

from copy import deepcopy as deepcopy

import os
import sys

import sys
# insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, '/Users/ppx/Desktop/PhD/rtdl')

from lib.deep import IndexLoader
import pandas as pd
import rtdl
from data import data as dta
import time



def apply_model(x_num, x_cat=None, model=None):
    if isinstance(model, rtdl.FTTransformer):
        return model(x_num, x_cat)
    elif isinstance(model, (rtdl.MLP, rtdl.ResNet)):
        assert x_cat is None
        return model(x_num)
    else:
        raise NotImplementedError(
            f'Looks like you are using a custom model: {type(model)}.'
            ' Then you have to implement this branch first.'
        )


@torch.no_grad()
def evaluate(part, model, X, y, y_std, task_type="regression"):
    model.eval()
    prediction = []

    batch_size = 1024
    permutation = torch.randperm(X[part].size()[0])

    for iteration in range(0, X[part].size()[0], batch_size):

        batch_idx = permutation[iteration:iteration + batch_size]

        x_batch = X[part][batch_idx]

        prediction.append(apply_model(x_batch, model=model))
    prediction = torch.cat(prediction).squeeze(1).cpu().numpy()
    target = y[part].cpu().numpy()

    if task_type == 'binclass':
        prediction = np.round(scipy.special.expit(prediction))
        score = sklearn.metrics.accuracy_score(target, prediction)
    elif task_type == 'multiclass':
        prediction = prediction.argmax(1)
        score = sklearn.metrics.accuracy_score(target, prediction)
    else:
        assert task_type == 'regression'
        score = sklearn.metrics.mean_squared_error(target, prediction) ** 0.5 * y_std
    return score



def learn_that(_model, _optimizer, _loss_fn, _X, _y, _epochs, _batch_size, _gse, _old_x, print_mode=False, _task_type="regression", sparse=False):

    start = time.time()
    if print_mode:
        print(f'Test score before training: {evaluate("test", _model):.4f}')

    report_frequency = len(_X['train']) // _batch_size // 5


    size = _X['train'].size()[0]
    column_count = len(_old_x['train'].columns)
    losses = {"val": [], "test": []}
    for epoch in range(1, _epochs + 1):

        print("epoch " + str(epoch) + " on " + str(_epochs) + " epochs \n")
        permutation = torch.randperm(size)

        for iteration in range(0, size, _batch_size):

            if iteration % 100 == 0:
                print(str(iteration) + " on " + str(size) + " iterations" )

            batch_idx = permutation[iteration:iteration + _batch_size]

            _model.train()
            _optimizer.zero_grad()
            x_batch = _X['train'][batch_idx]
            y_batch = _y['train'][batch_idx]

            loss = _loss_fn(apply_model(x_batch, model=_model).squeeze(1), y_batch)
            loss.backward()


            # Modify gradients
            if _gse:
                old_params = []
                for name, param in _model.named_parameters():
                    if name == "blocks.0.linear.weight":
                        factors = torch.ones(column_count,param.grad.shape[0])
                        for i in range(column_count):
                            idx = _old_x['train'][iteration * _batch_size:(iteration+1) * _batch_size].columns[i]
                            real_count = _old_x['train'][iteration * _batch_size:(iteration+1) * _batch_size][idx].sum()
                            if real_count > 0:
                                factors[i] = (_batch_size / (1.0 * real_count)) * factors[i]
                        param.grad = torch.mul(param.grad, torch.transpose(factors,0,1))
                        old_params.append(deepcopy(param))

            if sparse:
                for p in _model.parameters():
                    p.grad = p.grad.to_sparse()
                    
            _optimizer.step()
            if _gse and not sparse:
                i = 0
                for name, param in _model.named_parameters():
                    if name == "blocks.0.linear.weight":
                        param = torch.where(param.grad == 0, old_params[i], param)
                        i += 1
            if print_mode:
                if iteration % report_frequency == 0:
                    batch = "batch"
                    if _gse:
                        batch= "gse-batch"
                    print(f'(epoch) {epoch} ({batch}) {iteration} (loss) {loss.item():.4f}')

        losses['val'].append(float(_loss_fn(apply_model(_X['val'],   model=_model).squeeze(1), _y['val'])))
        losses['test'].append(float(_loss_fn(apply_model(_X['test'], model=_model).squeeze(1), _y['test'])))


    end = time.time()
    delta = end - start
    print("ellapsed time (sec) : " + str(delta))
    return losses
