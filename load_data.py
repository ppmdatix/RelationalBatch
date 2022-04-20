import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
from data import data as dta


def to_y(_data):
    if type(_data) == np.ndarray:
        return _data
    else:
        return _data.values


def load_data(path, target_name="target", task_type="multiclass", nrows=None, train_size=0.8):
    df = pd.read_csv(path, nrows=nrows)

    target = df[target_name].values
    target_values = list(set(target))

    target_mapping = {target_values[i] : i for i in range(len(target_values))}

    result = []
    for t in target:
        result.append(target_mapping[t])

    df_format = {"target": pd.DataFrame(result).values.reshape(df.shape[0]), "data": df.drop(target_name, axis=1) }

    assert task_type in ['binclass', 'multiclass', 'regression']

    X_all = df_format['data'].astype('float32')
    y_all = df_format['target'].astype('float32' if task_type == 'regression' else 'int64')
    if task_type != 'regression':
        y_all = sklearn.preprocessing.LabelEncoder().fit_transform(y_all).astype('int64')

    old_x = {}
    y = dict()
    old_x['train'], old_x['test'], y['train'], y['test'] = sklearn.model_selection.train_test_split(
        X_all, y_all, train_size=train_size
    )
    old_x['train'], old_x['val'], y['train'], y['val'] = sklearn.model_selection.train_test_split(
        old_x['train'], y['train'], train_size=train_size
    )

    # not the best way to preprocess features, but enough for the demonstration
    preprocess = sklearn.preprocessing.StandardScaler().fit(old_x['train'])
    X = {
        k: torch.tensor(preprocess.fit_transform(v), device=dta.device)
        for k, v in old_x.items()
    }
    y = {k: torch.tensor(to_y(v), device=dta.device) for k, v in y.items()}

    # !!! CRUCIAL for neural networks when solving regression problems !!!
    if task_type == 'regression':
        y_mean = y['train'].mean().item()
        y_std  = y['train'].std().item()
        y      = {k: (v - y_mean) / y_std for k, v in y.items()}
    else:
        y_std = None

    if task_type != 'multiclass':
        y = {k: v.float() for k, v in y.items()}

    return X, y, old_x, X_all, y_std, target_values

