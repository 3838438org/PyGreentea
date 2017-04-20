import time

import neuroglancer

from tasks.segmentation_and_evaluation import thresholds
from tasks.entities.datasets.test import affinity, truth
from tasks.segmentation_and_evaluation.seeding_and_segmenting import AffinityEvaluator

viewer = neuroglancer.Viewer(voxel_size=[8, 8, 8])

neuroglancer.set_static_content_source(url='https://neuroglancer-demo.appspot.com')
viewer.add(affinity, name='affinity')
viewer.add(truth, name='truth')


def add_to_viewer(results):
    try:
        viewer.add(results.fragments, name='mahotas fragments')
    except AttributeError:
        pass
    for idx, (s, t) in enumerate(zip(results.segmentations, results.thresholds)):
        # dostuff = (idx % 200) == 0
        dostuff = (t in (0.0, 0.8))
        if dostuff:
            viewer.add(s, "{} {}".format(results.method_name, t))


def make_figure(segmentors):
    import matplotlib.pyplot as plt
    figure, ax = plt.subplots()
    for segmentor, color in zip(segmentors, ['b', 'r', 'y']):
        segmentor.to_dataframe().plot(ax=ax, x="V_Info_merge", y="V_Info_split", kind="scatter",
                                      label=segmentor.method_name, c=color)
    return figure

if __name__ == "__main__":
    results = [
        # WaterzSegmentor(affinity, name="waterz normal disc=False"),
        # MahotasSegmentor(affinity, name="mahotas normal disc=False"),
        # WaterzSegmentor(affinity, name="waterz hist disc=False",
        #                 scoring_function="OneMinus<QuantileAffinity<AffinitiesType, 50, HistogramQuantileProviderSelect<256>::Value>>",
        #                 discretize_queue=False),
        # MahotasSegmentor(affinity, name="mahotas hist disc=False",
        #                 scoring_function="OneMinus<QuantileAffinity<AffinitiesType, 50, HistogramQuantileProviderSelect<256>::Value>>",
        #                 discretize_queue=False),
        # WaterzSegmentor(affinity, name="waterz normal disc=False", discretize_queue=True),
        # MahotasSegmentor(affinity, name="mahotas normal disc=False", discretize_queue=True),
        # WaterzSegmentor(affinity, name="waterz hist disc=True",
        #                 scoring_function="OneMinus<QuantileAffinity<AffinitiesType, 50, HistogramQuantileProviderSelect<256>::Value>>",
        #                 discretize_queue=True),
        # MahotasSegmentor(affinity, name="mahotas hist disc=True",
        #                  scoring_function="OneMinus<QuantileAffinity<AffinitiesType, 50, HistogramQuantileProviderSelect<256>::Value>>",
        #                  discretize_queue=True),
    ]
    for r in results:
        r.segment_and_evaluate(thresholds, groundtruth)
        add_to_viewer(r)
    figure = make_figure(results)
    figure.savefig("test.pdf")
    print(viewer)
    while True:
        time.sleep(1000)
