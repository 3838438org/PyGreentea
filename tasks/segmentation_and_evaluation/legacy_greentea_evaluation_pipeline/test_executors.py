from executors import SparkExecutor

spark_executor = SparkExecutor("test")

def funk(blah):
    import time
    time.sleep(5)
    print(blah)
    import os
    pid = os.getpid()
    print(pid)
    import socket
    hostname = socket.gethostname()
    filename = "/groups/turaga/home/grisaitisw/Desktop/spark-test-{}-{}-{}.txt".format(
        hostname, blah, pid
    )
    print(hostname)
    import waterz
    from pprint import pprint
    with open(filename, 'w') as f:
        for var in os.environ:
            f.write(var)
            f.write(os.environ.get(var, ""))
    return True


spark_executor.map(funk, range(50))
