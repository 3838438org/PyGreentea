import requests

good_components_cache = dict()


def load_good_components(uuid, exclude_strs, hostname='emdata2.int.janelia.org', port=7000):
    req = requests.get(
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


def get_good_components(uuid, name_substrings_to_exclude):
    args = (uuid, tuple(name_substrings_to_exclude))
    if args not in good_components_cache:
        good_components_cache[args] = load_good_components(uuid, name_substrings_to_exclude)
    return good_components_cache[args]
