import os
import sys


if len(sys.argv) == 1:
    datasets = ["kdd", "forest_cover", "adult_income", "dont_get_kicked", "used_cars", "compas"]
else:
    datasets = [sys.argv[1]]



models = ["mlp", "resnet"]
batch_sizes = [1, 8,32, 128]
epochs = 10
reproduction = 10


for dataset in datasets:
    if dataset.lower() == "kdd":
        task_type = "multiclass"
    elif dataset.lower() == "forest_cover":
        task_type = "multiclass"
    elif dataset.lower() == "adult_income":
        task_type = "binclass"
    elif dataset.lower() == "dont_get_kicked":
        task_type = "binclass"
    elif dataset.lower() == "used_cars":
        task_type = "regression"
    elif dataset.lower() == "compas":
        task_type = "binclass"
    for model in models:
        for batch_size in batch_sizes:
            command = "python aTOz.py %s %s %s %s %s %s" % (dataset, task_type, model, str(epochs),  str(batch_size), str(reproduction))
            print("On %s Launched model:%s - batch:%s" % (dataset, model, str(batch_sizes)))
            os.system(command)
            print("On %s Terminated model:%s - batch:%s" % (dataset, model, str(batch_sizes)))
