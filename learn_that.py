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
sys.path.insert(1, '/Users/ppx/Desktop/PhD/rtdl')

from lib.deep import IndexLoader
import pandas as pd
import rtdl




device = torch.device('cpu')


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


def none_if_small(x, epsilon=10e-4):
    if abs(x < epsilon):
        return None
    else:
        return  x
def small_if_none(x):
    if x is None:
        return 0.0
    else:
        return  x

def learn_that(_model, _optimizer, _loss_fn, _X, _y, y_std, _epochs, _batch_size, _relational_batch, _old_X, print_mode=False, _task_type="regression", sparse=False):
    # Docs: https://yura52.github.io/zero/reference/api/zero.data.IndexLoader.html


    batch_size = 256
    train_loader = IndexLoader(len(_X['train']), batch_size, device=device, shuffle=False)

    if print_mode:
        print(f'Test score before training: {evaluate("test", _model):.4f}')




    report_frequency = len(_X['train']) // _batch_size // 5
    losses = dict()
    losses['val'] = []
    losses['test'] = []
    for epoch in range(1, _epochs + 1):

        # X is a torch Variable
        permutation = torch.randperm(_X['train'].size()[0])

        for iteration in range(0, _X['train'].size()[0], batch_size):

            batch_idx = permutation[iteration:iteration + batch_size]
            # x_batch, y_batch = _X['train'][batch_idx], _y['train'][batch_idx]

            #for iteration, batch_idx in enumerate(train_loader):
            _model.train()
            _optimizer.zero_grad()
            x_batch = _X['train'][batch_idx]
            y_batch = _y['train'][batch_idx]

            loss = _loss_fn(apply_model(x_batch, model=_model).squeeze(1), y_batch)
            loss.backward()

            factors = dict()

            # Modify gradients
            if _relational_batch:
                oldParams = []
                for name, param in _model.named_parameters():
                    if name == "blocks.0.linear.weight":
                        column_count = len(_old_X['train'].columns)
                        factors = torch.ones(column_count,param.grad.shape[0])
                        for i in range(column_count):
                            column = _old_X['train'].columns[i]
                            if True:  # not column in oldNames:
                                idx = _old_X['train'][iteration * batch_size:(iteration+1) * batch_size].columns[i]
                                realCount = _old_X['train'][iteration * batch_size:(iteration+1) * batch_size][idx].sum()
                                if realCount > 0:
                                    factors[i] = (batch_size / (1.0 * realCount)) * factors[i]
                                else:
                                    ()
                                    # factors[i] = float('nan') * factors[i]
                        param.grad = torch.mul(param.grad, torch.transpose(factors,0,1))
                        oldParams.append(deepcopy(param))

            if optim == "sparse":
                for p in model.parameters():
                    p.grad = p.grad.to_sparse()
                    
            _optimizer.step()
            if _relational_batch:
                i = 0
                for name, param in _model.named_parameters():
                    if name == "blocks.0.linear.weight":
                        param = torch.where(param.grad == 0, oldParams[i], param)
                        i += 1
            if iteration % report_frequency == 0:
                batch = "batch"
                if _relational_batch:
                    batch= "relational-batch"
                if print_mode:
                    print(f'(epoch) {epoch} ({batch}) {iteration} (loss) {loss.item():.4f}')

        losses['val'].append(float(_loss_fn(apply_model(_X['val'],model=_model).squeeze(1), _y['val'])))
        losses['test'].append(float(_loss_fn(apply_model(_X['test'],model=_model).squeeze(1), _y['test'])))

        val_score  = evaluate("val", _model, _X, _y, y_std, task_type=_task_type)
        test_score = evaluate("test", _model, _X, _y, y_std, task_type=_task_type)

    return losses
