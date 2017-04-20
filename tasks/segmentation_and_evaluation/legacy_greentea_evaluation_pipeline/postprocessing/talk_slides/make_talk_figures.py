from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from results import results_name

sns.set_style('dark')


def make_figure(figure_spec):
    plot_kwargs = dict(
        x="v_rand_split",
        y="v_rand_merge",
        xlim=(0.84, 1.0),
        ylim=(0.84, 1.0),
        legend=False,
    )
    fig = plt.figure(**figure_spec["figure_kwargs"])
    ax = fig.add_subplot(1, 1, 1)
    for series_spec in figure_spec["series"]:
        df = series_spec["df"]
        series_plot_kwargs = plot_kwargs.copy()
        series_plot_kwargs.update(series_spec["plot_kwargs"])
        try:
            ax = df.plot(
                ax=ax,
                **series_plot_kwargs
            )
        except TypeError, e:
            print(e)
            print(series_spec["description"])
    ax.set_xlabel("VRand (split)")
    ax.set_ylabel("VRand (merge)")
    legend = ax.legend(
        tuple(series["description"] for series in figure_spec["series"]),
        **figure_spec["legend_kwargs"]
    )
    ax.grid("on")
    ax.set_title(**figure_spec["title_kwargs"])
    filename = "plots_{}_{}.pdf".format(figure_spec["name"], results_name)
    fig.savefig(filename, bbox_extra_artists=(legend,), bbox_inches='tight')
    return fig

from mb import figure_spec

fig = make_figure(figure_spec)

