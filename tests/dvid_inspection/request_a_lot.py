import multiprocessing as mp
import requests
import time

logger = mp.log_to_stderr()
logger.setLevel(20)


def request_dvid(session):
    logger.info("starting")
    # url = 'http://slowpoke1:22203/api/node/e402c09ddd0f45e980d9be6e9fcb9bd0/grayscale/raw/0_1_2/132_132_132/5617_1908_1823/nD'
    url = 'http://slowpoke1:22203/api/node/e402c09ddd0f45e980d9be6e9fcb9bd0/grayscale/info'
    response = session.get(url)
    logger.info("done")
    return 1


pool = mp.Pool(40)


applies = []

for _ in range(2000):
    session = requests.Session()
    applied = pool.apply_async(request_dvid, (session,))
    applies.append(applied)


for app in applies:
    app.get()
