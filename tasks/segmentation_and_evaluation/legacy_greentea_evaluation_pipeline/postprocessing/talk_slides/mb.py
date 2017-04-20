import numpy as np
import pandas as pd

from evaluation.postprocessing.talk_slides import make_series


toufiq_mb_4440 = pd.DataFrame(dict(
        v_info_merge=[np.NaN],
        v_info_split=[np.NaN],
        v_info=[np.NaN],
        v_rand_merge=[0.95195414660147826],
        v_rand_split=[0.99104899920813228],
        v_rand=[0.97110826229601066],
    ))
toufiq_mb_4400 = pd.DataFrame(dict(
        v_rand_merge=[0.86382846233749366],
        v_rand_split=[0.98854791473491133],
        v_rand=[0.92198954348791717],
    ))
toufiq_result = pd.concat([toufiq_mb_4440, toufiq_mb_4400])


figure_spec = dict(
    series=(
        make_series("run_0723_02", 70000, "4440", "multi-scale convnet + MALIS (train: MB 4400 only)"),
        make_series("run_0723_07", 200000, "4440", "multi-scale convnet + MALIS (train: small medulla)"),
        make_series("run_0723_11", 30000, "4440", "multi-scale convnet + MALIS (train: small medulla, PB, MB 4400)"),
        make_series("run_0723_12", 350000, "4440", "multi-scale convnet + MALIS (train: big medulla)"),
        dict(
            description="FlyEM pipeline model",
            df=toufiq_result,
            plot_kwargs=dict(
                style=".",
                color="Black",
                marker=(8, 1, 0),
                markersize=12
            ),
        ),
    ),
    title_kwargs=dict(
        label="VRand on MB 4440",
        y=1.05,
    ),
    figure_kwargs=dict(
        figsize=(6, 6),
    ),
    legend_kwargs=dict(
        loc="lower left",
        bbox_to_anchor=(-0.05, -0.45),
    ),
    name="mb_4440_2",
)
