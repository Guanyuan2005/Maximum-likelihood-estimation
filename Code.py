import numpy as np
from scipy.stats import weibull_min
import math
pai=3.1415926


if __name__=="__main__":
    n = 10000  # number of samples
    k = 0.2  # shape
    lam = 0.5  # scale
    x = weibull_min.rvs(k, loc=0, scale=lam, size=n)
    ln_x = np.log(x)
    ln_x_2 = ln_x * ln_x
    ln_x_3 = np.power(ln_x, 3)
    k_0 = pai / math.sqrt(6 * (np.sum(ln_x_2) / n - pow(np.sum(ln_x), 2) / (n * n)))
    x_k0 = np.power(x, k_0)
    x_k0_sum = np.sum(x_k0)
    x_c0 = (x_k0 / x_k0_sum - 1 / n) * ln_x
    c_0 = np.sum(x_c0)
    ln_x_3_x_k0 = ln_x_3 * x_k0
    ln_x_2_x_k0 = ln_x_2 * x_k0
    ln_x_1_x_k0 = ln_x * x_k0
    c_1 = np.sum(ln_x_2_x_k0) / x_k0_sum - pow(np.sum(ln_x_1_x_k0) / x_k0_sum, 2)
    c_2 = np.sum(ln_x_3_x_k0) / (2 * x_k0_sum) + pow(np.sum(ln_x_1_x_k0) / x_k0_sum, 3) - 3 * np.sum(
        ln_x_1_x_k0) * np.sum(ln_x_2_x_k0) / (2 * pow(x_k0_sum, 2))
    k_1 = (1 - k_0 * c_0) / (c_0 + c_1 * k_0)
    k_2 = -(k_1 * k_1 * (c_1 + c_2 * k_0) / (c_0 + c_1 * k_0))
    k_res = k_0 + k_1 + k_2
    x_k_res = np.power(x, k_res)
    lam_res = pow(np.sum(x_k_res) / n, 1 / k_res)
    print("k_0")
    print(round(k_0,5))
    print("k_1")
    print(round(k_1,5))
    print("k_2")
    print(round(k_2,7))
    print("k_res")
    print(round(k_res,5))
    print("lam_res")
    print(round(lam_res,5))