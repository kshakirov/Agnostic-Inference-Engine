
import random
from  inference_core.wasserman import calculate_empirical_moments
# Принудительное обновление при каждом C-c C-c




print("testing wasserman")


# Generate a list of 4 tuples, each containing an (x, y) coordinate pair
num_tuples = 10
sample  = [(True if random.randint(0,1) ==0 else False , random.randint(0, 10)) for _ in range(num_tuples)]


#sample = ((True, 1),(True, 1))
p_hat, variance, conf_interval = calculate_empirical_moments(sample)
print(p_hat,variance, conf_interval)
