import mahotas
import numpy as np


def get_seeds(boundary, method, next_id=1):
    if method == 'grid':
        height = boundary.shape[0]
        width = boundary.shape[1]
        seed_positions = np.ogrid[0:height:seed_distance, 0:width:seed_distance]
        num_seeds_y = seed_positions[0].size
        num_seeds_x = seed_positions[1].size
        num_seeds = num_seeds_x * num_seeds_y
        seeds = np.zeros_like(boundary).astype(np.int32)
        seeds[seed_positions] = np.arange(next_id, next_id + num_seeds).reshape((num_seeds_y, num_seeds_x))
    elif method == 'minima':
        minima = mahotas.regmin(boundary)
        seeds, num_seeds = mahotas.label(minima)
        seeds += next_id
        seeds[seeds == next_id] = 0
    elif method == 'maxima_distance':
        distance = mahotas.distance(boundary < 0.5)
        maxima = mahotas.regmax(distance)
        seeds, num_seeds = mahotas.label(maxima)
        seeds += next_id
        seeds[seeds == next_id] = 0
    else:
        raise ValueError("method not recognized")
    return seeds, num_seeds


def get_fragments(affinities):
    affs_xyz = 1.0 - (affinities[1] + affinities[2] + affinities[0]) / 3
    next_id = 1
    seeds, num_seeds = get_seeds(affs_xyz, "minima", next_id=next_id)
    fragments = mahotas.cwatershed(affs_xyz, seeds)
    return fragments.astype(np.uint64)


class MahotasFragmenter(object):
    name = 'mahotas'

    def __call__(self, affinity):
        return get_fragments(affinity)
