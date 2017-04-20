import numpy as np
import pandas as pd

from evaluation.postprocessing.talk_slides import make_series


figure_spec = dict(
    series=(
        make_series("fibsem17_srini_original_affs", 290000, "tstvol-520-2-h5", "fibsem17"),
        make_series("run_1202_2", 250000, "tstvol-520-2-h5", "run_1202_2"),
        make_series("run_1129_4", 130000, "tstvol-520-2-h5", "run_1129_4"),
        make_series("run_1111_2", 190000, "tstvol-520-2-h5", "run_1111_2"),
        make_series("run_0712_2", 330000, "tstvol-520-2-h5", "run_0712_2"),
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
    filename="plots_medulla_tstvol-520-2-h5.pdf",
)
