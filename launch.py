import os
import sys
from data import data as dta

import data.data

if len(sys.argv) == 1:
    datasets = dta.datasets
else:
    datasets = [sys.argv[1]]

models = dta.models
batch_sizes = dta.batch_sizes
epochs = dta.epochs
reproduction = dta.reproduction


for dataset in datasets:
    task_type = dta.task_types[dataset.lower()]
    for model in models:
        for batch_size in batch_sizes:
            command = "python aTOz.py %s %s %s %s %s %s" % (dataset, task_type, model, str(epochs), str(batch_size), str(reproduction))
            print("On %s Launched model:%s - batch:%s" % (dataset, model, str(batch_sizes)))
            os.system(command)
            print("On %s Terminated model:%s - batch:%s" % (dataset, model, str(batch_sizes)))
