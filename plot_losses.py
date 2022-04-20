import numpy as np
from matplotlib import pyplot as plt


def create_path(data_path, model, e, bs, rb):
    result = data_path           + "/"
    result += "model"            + str(model) + "-"
    result += "epochs"           + str(e) + "-"
    result += "batch-size"       + str(bs) + "-"
    result += "relational-batch" + str(rb)

    return result


def plot_losses(_losses, title=None, path=None, print_mode=False):
    for key in _losses:
        plt.plot([np.log(x) for x in _losses[key]], label=key)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    if title is not None:
        plt.title(title)

    if path is not None:
        plt.savefig(path)
    if print_mode:
        plt.show()
    plt.close()
