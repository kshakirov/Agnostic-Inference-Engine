from  inference_core.wasserman import calculate_empirical_moments
from inference_core.wald_test_conf_interval import wald_test, confidence_interval
import numpy as np
from math import erf,sqrt,floor,log
from scipy.stats import norm


from inference_core.utils import prep_data

def file_template(size,run="001"):
    return f"/Users/kiryloshakirov/Documents/ChatGPT/Agnostic-Inference-Engine/experiments/latency/sclerotix/post-body-matrix/body-{size}b-run-{run}.csv"

LENGTH = 1000

latency_vector = np.array([])

body_sizes = np.array([
    16, 32, 64, 128, 256, 512, 1024,
    2048, 4096, 8192, 16384, 32768, 65536,
])


def log_body_sizes(b_s):
    return np.log2(b_s)

def create_difff_matrix(b_s):
    column_sizes = np.repeat(b_s, LENGTH)
    return b_s[:,None] - column_sizes[None,:]


def create_weight_matrix(d_m, h):
    w_m = norm.pdf(d_m/h)
    normalized_weights = (
        w_m
        / w_m.sum(axis=1, keepdims=True)
    )
    return normalized_weights

def normalize_weight_matrix(w_m):
    normalized_weights = (
        w_m
        / w_m.sum(axis=1, keepdims=True)
    )
    return normalized_weights



def estimate_latency(w_m,l_v):
    return w_m @ l_v


def nullify_size_col_values(w_m, body_sizes,length):
    for i in range(body_sizes.size):
        start = i * length
        end = start + length
        w_m[i, start:end] = 0

    


for s in body_sizes:
    d = prep_data(file_template(s),LENGTH)
    latency_vector = np.concatenate([latency_vector,d])


assert(LENGTH * body_sizes.size == latency_vector.size)

body_sizes_log2 = log_body_sizes(body_sizes)
diff_matrix = create_difff_matrix(body_sizes_log2)

assert(diff_matrix.shape == (body_sizes.size,latency_vector.size))

w_m = create_weight_matrix(diff_matrix, 1)
assert(w_m.shape == (body_sizes.size,latency_vector.size))

estimated_latencies = estimate_latency(w_m, latency_vector)
assert(body_sizes.size == estimated_latencies.size)
#print(estimated_latencies)


#nullify_size_col_values(w_m, body_sizes_log2, LENGTH)


h_values = np.array([
    0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0
])

min_error=    float('inf')
best_h =0
real_means = latency_vector.reshape(
    body_sizes.size, LENGTH
).mean(axis=1)

for h in h_values:
    w_m_local = create_weight_matrix(diff_matrix, h)
    nullify_size_col_values(w_m_local, body_sizes, LENGTH)
    w_m_local = normalize_weight_matrix(w_m_local)
    estimated_latencies_local = estimate_latency(w_m_local, latency_vector)
    diffs = np.subtract(real_means, estimated_latencies_local)
    square_diffs =np.square(diffs)
    avg_square_diffs = np.mean(square_diffs)
    print(f" h = {h} mse = {avg_square_diffs} choosing the best ")
    if avg_square_diffs < min_error:
        min_error = avg_square_diffs
        best_h = h

print(f"best h = {best_h} with error {min_error}")
