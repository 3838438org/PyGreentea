import os

import pickle

from dvision import dvid_requester

good_components_cache = dict()


def request_good_components(uuid, exclude_strs, hostname='emdata1.int.janelia.org', port=7000):
    req = dvid_requester.get(
        url='http://{hostname}:{port}/api/node/{uuid}/annotations/key/annotations-body'.format(
            hostname=hostname,
            port=port,
            uuid=uuid,
        )
    )
    json_response = req.json()
    result = []
    for item in json_response['data']:
        if 'name' not in item:
            body_should_be_excluded = True
        else:
            body_should_be_excluded = any(text in item['name'] for text in exclude_strs)
        if not body_should_be_excluded:
            result.append((long(item['body ID'])))
    return result


def load_good_components(uuid, name_substrings_to_exclude):
    hash_value = hash(uuid + str(name_substrings_to_exclude))
    file_path = 'good_components_list.' + str(hash_value) + '.pickle'
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            result = pickle.load(f)
    else:
        result = request_good_components(uuid, name_substrings_to_exclude)
        with open(file_path, 'w') as f:
            pickle.dump(result, f)
    assert type(result) is list
    return result


def get_good_components(uuid, name_substrings_to_exclude):
    args = (uuid, tuple(name_substrings_to_exclude))
    if args not in good_components_cache:
        good_components_cache[args] = load_good_components(uuid, name_substrings_to_exclude)
    return good_components_cache[args]
