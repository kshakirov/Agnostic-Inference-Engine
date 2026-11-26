import numpy as np

from inference_core.estimation.linear_regression_tools import simple_linear_regression
from inference_core.utils import prep_data

x = np.array([1., 2., 3., 4.])
y = np.array([2., 3., 5., 4.])

def file_template(size,run="001"):
    return f"/Users/kiryloshakirov/Documents/ChatGPT/Agnostic-Inference-Engine/experiments/latency/sclerotix/post-body-matrix/body-{size}b-run-{run}.csv"


body_sizes = np.array([
    16, 32, 64, 128, 256, 512, 1024,
    2048, 4096, 8192, 16384, 32768, 65536,
])


LENGTH = 1000

latencies = []
for y_i in body_sizes:
    x_i = prep_data(file_template(y_i), LENGTH)
    latencies.append(np.average(x_i))
print(f"y={latencies}")
print(f"x = {body_sizes}")
x = body_sizes
y = np.array(latencies)
A,B = simple_linear_regression(x, y)

print(f"solution is {A}x + {B}")
