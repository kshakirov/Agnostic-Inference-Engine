import numpy as np
def prep_data(file_name, length):
    """ получить эпирические данные  from file"""
    np_data = np.zeros(length,dtype=np.float64)
    with open(file_name) as fd:
        for i, line in enumerate(fd):
            cells = line.split(',')
            np_data[i] = float(cells[2])/1000000
    return np_data
