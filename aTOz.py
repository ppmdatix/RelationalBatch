from load_data import load_data
from create_model import create_model
from learn_that import learn_that
from plot_losses import plot_losses, create_path
import sys
import pandas as pd
from box_plot import box_plot
from copy import deepcopy
from data import data as dta

dataset = sys.argv[1].lower()


nrows = None
optims = dta.optims

dataDir = "data/" + dta.folderName[dataset] + "/"
path    = dataDir + "training_processed.csv"
resDir  = "results/" + dta.folderName[dataset] + "/"
target  = dta.targets[dataset]



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

results = {"gse-"+o: [] for o in optims}
for o in optims:
    results["no_gse-"+o] = []


for _k in range(k):
    print(str(k) + "\n")
    for optim in optims:
        for gse in [True, False]:
            if gse:
                model, optimizer, loss_fn = create_model(X_all, n_classes=n_classes, task_type=task_type, model_name=model_name, optim=optim)
                modelGSE     = deepcopy(model)
                optimizerGSE = deepcopy(optimizer)
                loss_fnGSE   = deepcopy(loss_fn)
            else:
                model, optimizer, loss_fn = modelGSE, optimizerGSE, loss_fnGSE

            losses = learn_that(
                        model,
                        optimizer,
                        loss_fn,
                        X,
                        y, 
                        epochs,
                        batch_size,
                        gse,
                        old_x,
                        print_mode=False,
                        _task_type=task_type,
                        sparse=optim == "sparse_adam")
            if gse:
                results["gse-"+optim].append(losses["test"][-1])
            else:
                results["no_gse-"+optim].append(losses["test"][-1])

            if _k == 1:
                plot_path = create_path(resDir, model_name + "withOptimoo"+optim, epochs, batch_size, gse)
                title = dataset + "-gse:" + str(gse)
                plot_losses(losses, title=title, path=plot_path)
                df = pd.DataFrame(losses)
                df.to_csv(plot_path + '.csv', index=False)
if k > 1:
    save_path = create_path(resDir, model_name+ "withOptimoo"+optim, epochs, batch_size, k)
    print(results)
    box_plot(results, path=save_path)
