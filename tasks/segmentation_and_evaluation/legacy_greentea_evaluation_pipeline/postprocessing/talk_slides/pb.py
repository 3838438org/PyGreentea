import numpy as np
import pandas as pd

from evaluation.postprocessing.talk_slides import make_series


toufiq_pb2 = pd.DataFrame(dict(
        v_info_merge=[np.NaN],
        v_info_split=[np.NaN],
        v_info=[np.NaN],
        v_rand_merge=[0.97635296900795798],
        v_rand_split=[0.97223959383006053],
        v_rand=[0.97429193986098517],
    ))

figure_spec = dict(
    series=(
        make_series("run_0822_8", 370000, "pb2", "multi-scale convnet + MALIS (train: PB)"),
        make_series("run_0822_7", 190000, "pb2", "multi-scale convnet + MALIS (train: PB)"),
        make_series("run_0723_08", 80000, "pb2", "multi-scale convnet + MALIS (train: small medulla)"),
        make_series("run_0723_09", 140000, "pb2", "multi-scale convnet + MALIS (train: small medulla, PB, MB 4400)"),
        make_series("run_0712_2", 170000, "pb2", "multi-scale convnet + MALIS (train: big medulla)"),
        dict(
            description="FlyEM pipeline model",
            df=toufiq_pb2,
            plot_kwargs=dict(
                style=".",
                color="Black",
                marker=(8, 1, 0),
                markersize=12
            ),
        ),
    ),
    title_kwargs=dict(
        label="VRand on PB2",
        y=1.05,
    ),
    figure_kwargs=dict(
        figsize=(6, 6),
    ),
    legend_kwargs=dict(
        loc="lower left",
        bbox_to_anchor=(-0.05, -0.4),
    ),
    name="pb2",
)
