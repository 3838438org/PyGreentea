import numpy as np
import pandas as pd

from evaluation.postprocessing.talk_slides import make_series


flyem_result = pd.DataFrame(dict(
        v_info_merge=np.NaN,
        v_info_split=np.NaN,
        v_info=np.NaN,
        v_rand=[0.91649400174669404, 0.90723984977837013],
        v_rand_merge=[0.87446360985019356, 0.87653558376803919],
        v_rand_split=[0.96276870731274256, 0.94017328827602964],
    ))

flyem_series = dict(
    description="FlyEM pipeline model",
    df=flyem_result,
    plot_kwargs=dict(
        style=".",
        color="Black",
        marker=(8, 1, 0),
        markersize=12
    ),
)


figure_spec = dict(
    series=(
        make_series("fibsem17_srini_original_affs", 290000, "tstvol-520-2-h5", "fibsem17"),
        make_series("run_1202_2", 250000, "tstvol-520-2-h5", "run_1202_2"),
        make_series("run_1129_4", 130000, "tstvol-520-2-h5", "run_1129_4"),
        make_series("run_1111_2", 190000, "tstvol-520-2-h5", "run_1111_2"),
        make_series("run_0712_2", 310000, "tstvol-520-2-h5", "run_0712_2"),
        flyem_series,
    ),
    title_kwargs=dict(
        label="VRand on FIBSEM Medulla tstvol-520-2-h5",
        y=1.05,
    ),
    figure_kwargs=dict(
        figsize=(6, 6),
    ),
    legend_kwargs=dict(
        loc="lower left",
        bbox_to_anchor=(-0.05, -0.45),
    ),
    name="fibsem17_replicas",
)
