from  inference_core.wasserman import calculate_empirical_moments
import numpy as np
import array


def empiricall_cdf(sample):
    print(f"_empirically_cdf: caculating cdf from [{len(sample)}] ")
    x = np.sort(sample)
    y_direct = np.arange(len(x))
    y_rev = y_direct[::-1]
    y_i = np.divide(y_direct,y_rev)
    y= np.multiply(x,y_i)
    return y 
    
    

print("Followng the plan")
fd = open("/Users/kiryloshakirov/Documents/ChatGPT/Agnostic-Inference-Engine/experiments/latency/2026-08-31-baseline-a/baseline-a.csv")
print(fd)


np_data = np.zeros(600,dtype=np.float64)


for i, line in enumerate(fd):
    cells = line.split(',')
#    print(float(cells[]))
    np_data[i] = float(cells[2])/1000000
#    break
#data[:10]

np_data[0:10]
mean = np.mean(np_data)
# mean
var  = np.var(np_data)
std = np.std(np_data)

print(f"mean = {mean}\n var =  {var}\n std =  {std}")


empiricall_cdf(np_data)
