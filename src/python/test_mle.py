from inference_core.mle.generator import generate_normal_stream, mean_std




estms = []
for i in range(10):
    s = generate_normal_stream(1000, 150, 10000,i )
    m,std = mean_std(s)
    print(m,std)
    estms.append((m,std))


means = [m for (m,std) in estms]
stds = [std for (m,std) in estms]

params_of_means = mean_std(means)
params_of_stds = mean_std(stds)

print(f"params of means {params_of_means}")
print(f"params of stds {params_of_stds}")
