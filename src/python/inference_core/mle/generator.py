import numpy as np

def generate_normal_stream(mu,sigma,size):
    """ generating normal stream """
    return  np.random.normal(loc=mu, scale=sigma, size=size)

def add_noise(data, mu, level):
    """ adding noise """
    noise_abs = level * mu    
    noisy_data = np.random.normal(0, noise_abs / 2, len(data))
    return np.add(data, noisy_data)                                  

        
    


stream = generate_normal_stream(1000, 150, 1000)
stream[0:10]

stream = add_noise(stream, 1000, 0.05)
stream[0:10]
