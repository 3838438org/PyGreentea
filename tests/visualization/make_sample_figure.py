from functools import reduce
from operator import mul

import numpy as np

import visualization

image_shape = (8, 100, 100)
truth_shape = (6, 80, 80)
pred_shape = (4, 40, 40)
random_state = np.random.RandomState(seed=0)

dataset = dict(
    data=random_state.randint(0, 256, reduce(mul, image_shape)).reshape(image_shape),
    label=random_state.randint(0, 2, 3 * reduce(mul, truth_shape)).reshape((3,) + truth_shape),
    components=random_state.randint(0, 5, reduce(mul, truth_shape)).reshape(truth_shape),
    pred=random_state.uniform(0, 1, 3 * reduce(mul, pred_shape)).reshape((3,) + pred_shape),
    predseg=random_state.randint(0, 5, reduce(mul, pred_shape)).reshape(pred_shape),
)

z = 2

f = visualization.showme(dataset, z)

f.savefig('test.png')
