from matplotlib import pyplot as plt
from data import data as dta


def box_plot(losses, title=None, path=None, fontsize="medium"):
    labels = losses.keys()
    colors = dta.colors

    final_losses = [losses[label] for label in labels]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    plt.xticks(rotation=45)

    bplot = ax.boxplot(final_losses,
                         vert=True,
                         patch_artist=True,
                         labels=labels)

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    ax.yaxis.grid(True)
    ax.set_ylabel('Final Loss')
    plt.legend(fontsize=fontsize)
    if title is not None:
        plt.title(title)

    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()
