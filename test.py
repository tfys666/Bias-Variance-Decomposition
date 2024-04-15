import numpy as np
import pandas as pd
from bias_variance import sim


data1 = np.array([[100, 100, 100],[1, 2, 15]])
for i in range(len(data1)+1):
    re = sim(data1[0, i], data1[1, i])
    re_pd = pd.DataFrame(re,
                         index=["bias2", "var", "var_eps", "err"],
                         columns=["p=1", "p=2", "p=3"])
    print(f"n, sigma = {data1[0, i]}, {data1[1, i]}")
    print(re_pd)


data2 = np.array([[10, 200, 5000],[3, 3, 3]])
for i in range(len(data2)+1):
    re = sim(data2[0, i], data2[1, i])
    re_pd = pd.DataFrame(re,
                         index=["bias2", "var", "var_eps", "err"],
                         columns=["p=1", "p=2", "p=3"])
    print(f"n, sigma = {data2[0, i]}, {data2[1, i]}")
    print(re_pd)