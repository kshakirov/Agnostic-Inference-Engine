from  inference_core.wasserman import calculate_empirical_moments
import numpy as np
import array
import matplotlib.pyplot as plt
from math import erf,sqrt


def empirical_cdf(sample):
    #print(f"_empirically_cdf: caculating cdf from [{len(sample)}] ")
    x = np.sort(sample)
    y_direct = np.arange(1, len(x) +1 )
    y_len = np.full(len(x),len(x))
    y_i = np.divide(y_direct,y_len)
#    y= np.multiply(x,y_i)
    return x,y_i

def normal_cdf_at_point(x, mu_hat, sigma_hat):
    z = (x - mu_hat) /sigma_hat
    return 0.5 * ( 1 + erf(z/sqrt(2)))

def partial_cdf_at_point(mu_hat, sigma_hat):
    def ncdf_at_point(x):
        z = (x - mu_hat) /sigma_hat
        return 0.5 * ( 1 + erf(z/sqrt(2)))
    return ncdf_at_point

def plot_empirical_and_normal_pdf(sample, mu_hat, sigma_hat, output_file):
    x_min = min(np.min(sample), mu_hat - 4 * sigma_hat)
    x_max = max(np.max(sample), mu_hat + 4 * sigma_hat)
    x_grid = np.linspace(x_min, x_max, 2000)
    normal_pdf = (
        np.exp(-0.5 * ((x_grid - mu_hat) / sigma_hat) ** 2)
        / (sigma_hat * np.sqrt(2 * np.pi))
    )

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.hist(
        sample,
        bins="fd",
        density=True,
        alpha=0.55,
        label="Empirical latency",
    )
    axis.plot(x_grid, normal_pdf, linewidth=2, label="Fitted Normal PDF")
    axis.axvline(mu_hat, linestyle="--", label="mean")
    axis.axvline(mu_hat - sigma_hat, linestyle=":", label="mean - std")
    axis.axvline(mu_hat + sigma_hat, linestyle=":", label="mean + std")
    axis.set_xlabel("Latency, ms")
    axis.set_ylabel("Density")
    axis.set_title("Empirical latency density vs fitted Normal model")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_file, dpi=160)
    plt.close(figure)

def bootstrap_normal_step(mu,sigma, length):
    rng = np.random.default_rng()
    np_array = rng.normal(loc=mu, scale=sigma, size=length)
    x_cdf,y_after_cdf= empirical_cdf(np_array)
    m = np.mean(np_array)
    s = np.std(np_array)
    ncdf_at_point = partial_cdf_at_point(m, s)
    vectorized_func = np.vectorize(ncdf_at_point)
    n_x = vectorized_func(x_cdf)
    d_after = np.max(np.abs(np.subtract(n_x, y_after_cdf)))
    y_before = np.arange(length) *  1/length
    d_before = np.max(np.abs(np.subtract(n_x, y_before)))
    #print(f"D before {d_before} D after {d_after}")
    d_star= max(d_before,d_after)
#    print(f"D real is {d_real}")
    return d_star
    
def bootstrap_normal(size, mu,sigma,length):
    d_s = np.full(size,0.0)
    for i in range(size):
        d_s[i] = bootstrap_normal_step(mu, sigma, length)
    return d_s

FILE_NAME = "/Users/kiryloshakirov/Documents/ChatGPT/Agnostic-Inference-Engine/experiments/latency/2026-08-31-baseline-a/baseline-a.csv"
LENGTH = 600
BOOTSTRAP_SIZE=1000

def prep_data(file_name, length):
    np_data = np.zeros(length,dtype=np.float64)
    with open(file_name) as fd:
        print(fd)
        for i, line in enumerate(fd):
            cells = line.split(',')
            np_data[i] = float(cells[2])/1000000
    return np_data

np_data = prep_data(FILE_NAME, LENGTH)
mean = np.mean(np_data)

var  = np.var(np_data)
std = np.std(np_data)

print(f"mean = {mean}\n var =  {var}\n std =  {std}")


x, y = empirical_cdf(np_data)
print(f"ecdf x 10  = {x[0:10]} y 10 = {y[0:10]}")

ncdf_at_point = partial_cdf_at_point(mean, std)


assert(normal_cdf_at_point(mean - std/2, mean, std) == ncdf_at_point(mean - std/2))

vectorized_func = np.vectorize(ncdf_at_point)

n_x = vectorized_func(x)

d_after_array = np.abs(np.subtract(n_x, y))
d_after = np.max(np.abs(np.subtract(n_x, y)))

y_before = np.arange(LENGTH) *  1/LENGTH

d_before_array = np.abs(np.subtract(n_x, y_before))
d_before = np.max(np.abs(np.subtract(n_x, y_before)))

print(f"D before {d_before} D after {d_after}")

d_real = max(d_before,d_after)
print(f"D real is {d_real}")


d_star_one = bootstrap_normal_step(mean, std, LENGTH)
print(f" D star one  is {d_star_one}")
d_star_s =bootstrap_normal(BOOTSTRAP_SIZE, mean, std, LENGTH)
d_star_max = np.max(d_star_s)

print(f" D star max is {d_star_max}")




# plot_empirical_and_normal_pdf(
#     np_data,
#     mean,
#     std,
#     "/Users/kiryloshakirov/Documents/ChatGPT/Agnostic-Inference-Engine/experiments/latency/2026-08-31-baseline-a/baseline-a-histogram-vs-normal.png",
# )


#
ro = (1 + len([d for d in d_star_s if d >= d_real])) / (BOOTSTRAP_SIZE + 1)
print(f"ro is {ro}")

print("Looking for index of D real max")

print("which d_after or d_before won")

ks_delta_i = None
ks_empirical_latency = None
ks_empirical_freq = None
ks_model_freq = None
if d_before >= d_after:
    indices = np.where(d_before_array >= d_real )[0];
    print(f"d_before array max indices {indices}")
    if len(indices) > 0:
        i = indices[0]
        ks_empirical_latency =  x[i]
        ks_empirical_freq =  y_before[i]
        ks_model_freq = n_x[i]
        ks_delta_i = ks_model_freq - ks_empirical_freq

else:
    indices = np.where(d_after_array >= d_real)[0];
    print(f"d_after array max indices {indices}")
    if len(indices) > 0:
        i = indices[0]
        ks_empirical_latency =  x[i]
        ks_empirical_freq =  y[i]
        ks_model_freq = n_x[i]
        ks_delta_i = ks_model_freq - ks_empirical_freq
        
    

print(f" KS non parametric criterion empirical latency {ks_empirical_latency}, empirical freq {ks_empirical_freq}, model freq {ks_model_freq}, delta is {ks_delta_i} ")


def quantile(cdf_x,cdf_y,y_value):
    if(y_value <= 1 and y_value >=0):
        y_i = np.where(cdf_y >= y_value)[0][0]
        return cdf_x[y_i]
    else:
        raise ValueError("quantile  value is incorrect must be 0 <= value <= 1")
    
    
print(f" quantiles: mediana {quantile(x, y, 0.5)} vs mu {mean} ,0.90  {quantile(x, y, 0.9)}, 0.95  {quantile(x, y, 0.95)}, 0.99  {quantile(x, y, 0.99)}")

#quantile(x, y, 2)

def outliers(cdf_x,cdf_y):
    y_i = np.where(cdf_y >= 0.99)[0][0]
    return cdf_x[y_i:]


def outliers_env(cdf_x,cdf_y, data):
    ots = outliers(cdf_x, cdf_y)
    for o in ots:
        indx = np.where(data ==o)[0][0]
        l_i = indx - 5
        r_i = indx + 5
        print(f"{o} index in data is {indx} env is {data[l_i:r_i]}")

outliers_env(x, y, np_data)
