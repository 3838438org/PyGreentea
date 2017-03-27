import ipyparallel

ipp_client = ipyparallel.Client(
    url_file="/groups/turaga/home/grisaitisw/.ipython/profile_greentea/security/ipcontroller-client.json",
    timeout=60 * 60  # 1 hour
)
executor = ipp_client.load_balanced_view()
executor.set_flags(retries=100000)
