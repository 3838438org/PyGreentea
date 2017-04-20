from evaluation.postprocessing.talk_slides.results import df_all_results

from models.model_datasets import training_sets, partial_training_sets

data_colors = {
    "pb": "Magenta",
    "tstvol-520-1-h5": "LimeGreen",
    "4400": "Blue",
}

model_colors = dict()

for data_name in data_colors:
    for model in training_sets[data_name]:
        model_colors[model] = data_colors[data_name]
for data_name in partial_training_sets:
    for model in partial_training_sets[data_name]:
        model_colors[model] = "brown"
for fib25_model in ("run_0712_2", "run_0723_12", "run_0723_13"):
    model_colors[fib25_model] = "DarkGreen"
for key, value in model_colors.iteritems():
    print(key, value)


def make_series(model, iteration, data_name, description=None, style=None, color=None):
    if description is None:
        description = "{} @ {}".format(model, iteration)
    query_text = "data_name in ('{dn}') and model == '{m}' and iteration == {i}"\
        .format(dn=data_name, m=model, i=iteration)
    df = df_all_results.query(query_text)
    return dict(
        description=description,
        df=df,
        plot_kwargs=dict(
            style=style or "-",
            color=color or model_colors[model]
        ),
    )
