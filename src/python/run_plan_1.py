from  inference_core.wasserman import calculate_empirical_moments
import numpy as np
import array


def empirical_cdf(sample):
    print(f"_empirically_cdf: caculating cdf from [{len(sample)}] ")
    x = np.sort(sample)
    y_direct = np.arange(1, len(x) +1 )
    y_len = np.full(len(x),len(x))
    y_i = np.divide(y_direct,y_len)
#    y= np.multiply(x,y_i)
    return x,y_i
    
    

file_name = "/Users/kiryloshakirov/Documents/ChatGPT/Agnostic-Inference-Engine/experiments/latency/2026-08-31-baseline-a/baseline-a.csv"
length = 600

def prep_data(file_name, length):
    np_data = np.zeros(length,dtype=np.float64)
    with open(file_name) as fd:
        print(fd)
        for i, line in enumerate(fd):
            cells = line.split(',')
            np_data[i] = float(cells[2])/1000000
    return np_data

np_data = prep_data(file_name, length)
mean = np.mean(np_data)

var  = np.var(np_data)
std = np.std(np_data)

print(f"mean = {mean}\n var =  {var}\n std =  {std}")


x, y = empirical_cdf(np_data)
print(f"ecdf x 10  = {x[0:10]} y 10 = {y[0:10]}")
