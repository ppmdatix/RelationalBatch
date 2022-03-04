from load_data import load_data
from create_model import create_model
from learn_that import learn_that
from plot_losses import plot_losses, create_path
import sys
import pandas as pd
from box_plot import box_plot
from copy import deepcopy



dataset = sys.argv[1]

nrows = None

optims = ["adam", "adagrad"]


if dataset.lower() == "kdd":
    dataDir = "data/KDD99/"
    path = dataDir + "training_processed.csv"# "fetch_kddcup99.csv"
    resDir = "results/KDD99/"
    target = "labels"
elif dataset.lower() == "forest_cover":
    dataDir = "data/Forest_Cover/"
    path = dataDir + "training_processed.csv"# "forest_cover.csv"
    resDir = "results/Forest_Cover/"
    target = "Cover_Type"
elif dataset.lower() == "adult_income":
    dataDir = "data/Adult_Income/"
    path = dataDir + "training_processed.csv"# "forest_cover.csv"
    resDir = "results/Adult_Income/"
    target = "target"
elif dataset.lower() == "dont_get_kicked":
    dataDir = "data/Dont_Get_Kicked/"
    path = dataDir + "training_processed.csv"# "forest_cover.csv"
    resDir = "results/Dont_Get_Kicked/"
    target = "target"
    nrows = 10000
elif dataset.lower() == "used_cars":
    dataDir = "data/Usedcarscatalog/"
    path = dataDir + "training_processed.csv"# "forest_cover.csv"
    resDir = "results/Usedcarscatalog/"
    target = "price_usd"
elif dataset.lower() == "compas":
    dataDir = "data/compas/"
    path = dataDir + "training_processed.csv"#
    resDir = "results/compas/"
    target = "is_recid"



else:
    raise Exception('no such dataset')

task_type  = sys.argv[2]
model_name = sys.argv[3]
epochs     = int(sys.argv[4])
batch_size = int(sys.argv[5])
k          = int(sys.argv[6])

target_name = "target"
if len(sys.argv) > 7:
    target_name = sys.argv[7]




X, y, old_x, X_all, y_std, target_values = \
    load_data(path, task_type=task_type, target_name=target_name, nrows=nrows)

if task_type == "multiclass":
    n_classes = len(target_values)
else:
    n_classes = None

results = {} # "rb": [], "norb": []}
for optim in optims:
    results["rb-"+optim] = []
    results["norb-"+optim] = []


for _k in range(k):
    print(str(k) + "\n")
    for optim in optims:
        for relational_batch in [True, False]:

            if relational_batch:

                model, optimizer, loss_fn = create_model(X_all, n_classes=n_classes, task_type=task_type, model_name=model_name, optim=optim)
                modelRB     = deepcopy(model)
                optimizerRB = deepcopy(optimizer)
                loss_fnRB     = deepcopy(loss_fn)
            else:
                model, optimizer, loss_fn = modelRB, optimizerRB, loss_fnRB

            losses = learn_that(
                        model,
                        optimizer,
                        loss_fn,
                        X,
                        y,
                        y_std,
                        epochs,
                        batch_size,
                        relational_batch,
                        old_x,
                        print_mode=False,
                        _task_type=task_type)
            if relational_batch:
                results["rb-"+optim].append(losses["test"][-1])
            else:
                results["norb-"+optim].append(losses["test"][-1])
            title = dataset + "-relationalBatch:" + str(relational_batch)
            if _k == 1:
                plot_path = create_path(resDir, model_name + "withOptimoo"+optim, epochs, batch_size, relational_batch)
                plot_losses(losses, title=title, path=plot_path)

                df = pd.DataFrame(losses)

                df.to_csv(plot_path + '.csv', index=False)
if k > 1:
    save_path = create_path(resDir, model_name+ "withOptimoo"+optim,epochs, batch_size, k)
    print(results)
    box_plot(results, path=save_path)
