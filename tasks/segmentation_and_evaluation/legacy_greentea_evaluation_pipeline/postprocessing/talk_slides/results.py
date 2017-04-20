import pandas as pd


# results_name = "df_big_0908_10"
results_name = "df_big_1210_1"
# results_name = "df_big_1207_1_eval0904_zw0.11"
# results_name = "df_big_1207_2_waterz_2016-12-02"


df_path = "/groups/turaga/home/grisaitisw/src/greentea-evaluation-pipeline/" \
          "evaluation/postprocessing/{}.csv".format(results_name)
dtypes = dict(
    iteration=int,
    threshold=float,
    data_name=str,
    description=str,
)
df_all_results = pd.read_csv(df_path, sep=",", dtype=dtypes)

def is_test_result(model, data_name):
    from models.model_datasets import training_sets, partial_training_sets
    is_a_training_eval = model in training_sets.get(data_name, [])
    is_a_partial_training_eval = is_a_training_eval or model in partial_training_sets.get(data_name, [])
    return not is_a_partial_training_eval


if "is test_result" not in df_all_results.columns:
    df_all_results["is_test_result"] = \
        df_all_results.apply(
            lambda row: is_test_result(row["model"], row["data_name"]),
            axis=1)
