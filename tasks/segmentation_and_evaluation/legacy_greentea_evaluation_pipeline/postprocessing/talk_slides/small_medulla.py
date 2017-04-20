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
        # make_series("fibsem32_srini_original_affs", 140000, "tstvol-520-2-h5", "fibsem32"),
        # make_series("fibsem17_srini_original_affs", 290000, "tstvol-520-2-h5", "fibsem17"),
        # make_series("run_1111_2", 190000, "tstvol-520-2-h5", "run_1111_2"),
        # make_series("run_0712_2", 310000, "tstvol-520-2-h5", "run_0712_2"),
        # make_series("run_0723_10", 190000, "tstvol-520-2-h5", "run_0723_10"),
        make_series("run_0723_16", 20000, "tstvol-520-2-h5", "run_0723_16"),
        make_series("fibsem32_srini_original_affs", 140000, "tstvol-520-2-h5"),
        make_series("fibsem17_srini_original_affs", 140000, "tstvol-520-2-h5"),
        make_series("run_1202_2", 270000, "tstvol-520-2-h5"),
        # make_series("run_0723_10", 190000,"tstvol-520-2-h5", color="#003333"),
        # make_series("run_0723_10", 240000,"tstvol-520-2-h5", color="#006666"),
        make_series("run_0723_10", 260000,"tstvol-520-2-h5", color="#009999"),
        make_series("run_0723_10", 280000,"tstvol-520-2-h5", color="#00bbbb"),
        # make_series("run_0723_10", 360000,"tstvol-520-2-h5", color="#00eeee"),
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
    name="small_medulla",
)
