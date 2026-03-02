from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from general_utils.utils import compute_samples_sizes, load_npz_as_dict

def truncate_colormap(cmap, minval=0.1, maxval=0.9, n=256):
    return LinearSegmentedColormap.from_list(
        f"trunc({cmap.name})",
        cmap(np.linspace(minval, maxval, n))
    )


"""
plot_subsamples
Plots RSA similarity time courses obtained from trial subsampling, for multiple
model layers arranged in a grid of subplots.

For each layer, the function loads precomputed subsampling results, plots all
subsample sizes using a color gradient, and overlays the full-trial similarity
in black. A single shared legend and common axes are used across subplots.

INPUT:
    - layers: list[str]
        Model layer names to plot (one subplot per layer).
    - cfg: Cfg
        Configuration object. Must define at least:
        new_fs, step_samples, max_size, n_iter, monkey_name,
        date, brain_area, model_name, img_size, n_trials.
    - ylim: tuple[float, float] | None (default: None)
        Optional y-axis limits applied to all subplots.

OUTPUT:
    - None
        Displays the figure.
"""
def plot_subsamples(paths, layers, cfg, ylim = None, save=False):
    n_layers = len(layers)
    ncols = 4
    nrows = int(np.ceil(n_layers / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4*ncols, 3*nrows),
        sharex=True, sharey=True
    )
    axes = axes.flatten()

    base_cmap = plt.cm.turbo
    cmap = truncate_colormap(base_cmap, 0.10, 0.90)

    n_samples = compute_samples_sizes(cfg)

    for ax, l in zip(axes, layers):
        file_name = (
            f"subsampling_{cfg.new_fs}Hz_{cfg.step_samples}-{cfg.max_size}_"
            f"{cfg.n_iter}iter_{cfg.monkey_name}_{cfg.date}_"
            f"{cfg.brain_area}_{cfg.RDM_metric}_{cfg.model_name}_{cfg.img_size}_{l}.npz"
        )
        results_path = f"{paths['livingstone_lab']}/tiziano/results/{file_name}"
        iter_dict = load_npz_as_dict(results_path)

        for idx, k in enumerate(n_samples):
            color = cmap(idx / (len(n_samples) - 1))
            ax.plot(iter_dict[str(k)].T, color=color, alpha=0.6)
            ax.set_title(l, fontsize=12)
            if l == layers[-1]:
                ax.plot(iter_dict[str(k)][0,:].T, color=color, alpha=0.6, label=f"{k} trials")

        
        if l == layers[-1]:
            ax.plot(
                iter_dict[str(cfg.n_trials)],
                color="black",
                linewidth=2,
                label="all trials"
            )
        else:
            ax.plot(
                iter_dict[str(cfg.n_trials)],
                color="black",
                linewidth=2,
            )

    # remove unused axes
    for ax in axes[len(layers):]:
        ax.axis("off")

    fig.supxlabel("Time from image onset (ms)", fontsize=20)
    fig.supylabel("RSA Similarity", fontsize=20)
    fig.suptitle(f"Brain area: {cfg.brain_area}", fontsize=27)
    fig.tight_layout()
    fig.legend(fontsize=13, bbox_to_anchor=(0.98, 0.90))
    # make space on the right for the legend
    fig.tight_layout(rect=[0.03, 0, 0.85, .97])
    # ax.set_xticks(xtickspos)           # positions of ticks
    # ax.set_xticklabels([int(xt*1000/cfg.new_fs) for xt in xtickspos], fontsize=15)
    if ylim is not None:
        for ax in axes[:n_layers]:
            ax.set_ylim(ylim[0], ylim[1])
    xtickspos = range(0, len(iter_dict[str(cfg.n_trials)])+1, 10)       
    for ax in axes[:n_layers]:  # only the used axes
        ax.set_xticks(xtickspos)
        ax.set_xticklabels([int(xt*1000/cfg.new_fs) for xt in xtickspos], fontsize=15)
        ax.tick_params(axis='both', labelsize=15)
    if save==True:
        return fig, axes
    else:
        plt.show()
