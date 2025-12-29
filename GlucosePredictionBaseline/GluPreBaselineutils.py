import pandas as pd
import numpy as np

def load_data(filename,maxlen=3456):
    data = pd.read_csv(filename)
    data = data[[str(x) for x in range(maxlen)]]
    data = np.array(data)
    return data