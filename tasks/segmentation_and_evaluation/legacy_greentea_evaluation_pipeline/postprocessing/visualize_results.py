import pandas as pd

from models.model_datasets import partial_training_sets, training_sets

df_all_results = pd.read_csv(
    "/groups/turaga/home/grisaitisw/src/greentea-evaluation-pipeline/evaluation/postprocessing/df_big_1111_2.csv",
    sep=","
)

# print(df_all_results.head())

print(len(df_all_results["model"].str.contains('run_1031_|run_11')))

is_a_recent_model = df_all_results["model"].str.contains('run_1031_|run_11')

print(df_all_results["model"].where(is_a_recent_model).value_counts(dropna=False))

columns_to_show = ["data_name", "model", "iteration", "threshold", "is_a_partial_training_eval",
                   "v_rand_merge", "v_rand_split",
                   "v_rand",
                    ]
best_v_rands = df_all_results.where(df_all_results["model"].str.contains('run_0923|run_1013')).groupby(["data_name", "is_a_partial_training_eval", "model"])['v_rand'].transform(max) == df_all_results['v_rand']
df_best_v_rand_models = df_all_results[best_v_rands][columns_to_show]


line_colors = list('bryk')
model_substrs = [
    "fibsem32_srini_original_affs",
#     "run_0912_1",
#     "run_0912_1",
#     "run_0912_1",
#     "run_0712_2",
#     "run_0916_1",
#     "run_0908_4",
#     "run_0912_3",
#     "run_0723_12",
    "run_0923_1",
#     "run_0923_2",
#     "run_0923_3",
#     "run_1013_1",
#     "run_1013_2",
]

x = df_all_results    [df_all_results['model'].str.contains('|'.join(model_substrs))]
x = x[df_all_results['data_name'].str.contains("tstvol-520-2-h5")]
x = x.pivot_table(
        values="v_rand",
        index="iteration",
        columns=("model", "data_name"),
        aggfunc=max,
        dropna=False,
    )
ax = x.plot(
        ylim=(0.5, 1),
        xlim=(0, 400000),
        grid=True,
        style=(['-%s' % c for c in line_colors] or '-'),
    )
ax = df_all_results    [df_all_results['model'].str.contains('|'.join(model_substrs))]    [df_all_results['data_name'].str.contains("tstvol-520-1-h5")]    .pivot_table(
        values="v_rand",
        index="iteration",
        columns=("model", "data_name"),
        aggfunc=max,
        dropna=False
    )\
    .plot(
        ax=ax,
        style=(['.--%s' % c for c in line_colors] or '.--'),
        grid=True,
        ylim=(0.5, 1),
        xlim=(0, 1000000),
        figsize=(12, 6),
    )

