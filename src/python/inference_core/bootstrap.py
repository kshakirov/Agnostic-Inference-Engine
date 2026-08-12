import random
import math
def get_metrics(filename):
    with open(filename, "rt") as fd:
        metrics = fd.readlines()
        tuples = []
        print(f"sample size is {len(metrics)}")
        for m in metrics:
           status, latency =  m.split(",")
           tuples.append((True if status == "True" else False, float(latency.strip())))
        return tuples


def bootstrap_naive(n, samples, cf):
    b_samples = []
    for i in range(0,n):
        b_sample = []
        for j in range(0,10000):
            r = math.floor((10000 * random.random()))
            b_sample.append(samples[r])
        r = cf(b_sample)
        b_samples.append(r)
    return b_samples

#b_samples = bootstrap_naive(2, metrics[0:10000])
            
        


