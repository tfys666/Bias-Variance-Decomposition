import numpy as np
from scipy import linalg
from scipy.stats import uniform, norm
np.set_printoptions(precision=4, suppress=True)


def _solve_sym(xtx, xty):
    mat = linalg.cholesky(xtx)
    return linalg.lapack.dpotrs(mat, xty)[0]

def ols(x, y):
    x_mean = x.mean(axis=0)
    y_mean = y.mean()
    x_scale = x - x_mean
    y_scale = y - y_mean
    x_scale_t = x_scale.T
    xtx = np.dot(x_scale_t, x_scale)
    xty = np.dot(x_scale_t, y_scale)
    b_scale = _solve_sym(xtx, xty)
    b_inter = y_mean - np.inner(x_mean, b_scale)
    return b_inter, b_scale

def _sim_ols_p(n, sigma, p, xs):
    x = uniform.rvs(loc=-1, scale=2, size=n)
    epsilon = norm.rvs(scale=sigma, size=n)
    y = 2 * np.exp(x) + epsilon
    z = np.repeat(x[:, np.newaxis], p, axis=1)
    for i in np.arange(1, p):
        z[:, i] = z[:, i] * z[:, i - 1]
    b_result = [ols(z[:, :j], y) for j in np.arange(1, p + 1)]
    b_result_0 = [b_inter for b_inter, _ in b_result]
    b_result_1 = [b_scale for _, b_scale in b_result]

    zs = np.repeat(xs[:, np.newaxis], p, axis=1)
    for i in np.arange(1, p):
        zs[:, i] = zs[:, i] * zs[:, i - 1]
    zs_p = [zs[:, :j] for j in np.arange(1, p + 1)]
    re = np.array(list(map(np.dot, zs_p, b_result_1))) + \
         np.array(b_result_0)[:, np.newaxis]
    return re

def sim(n, sigma, p=3, n1=10000, k=1000):
    xs = uniform.rvs(loc=-1, scale=2, size=n1)
    epsilon = norm.rvs(scale=sigma, size=n1)
    f_xs = 2 * np.exp(xs)
    ys = f_xs + epsilon
    re = [_sim_ols_p(n, sigma, p, xs) for _ in np.arange(k)]
    re = np.array(re)
#期望误差
    err = re - ys
    err = err * err
    err = np.mean(err, axis=(0, 2))
#偏差
    re_mean = re.mean(axis=0)
    bias = re_mean - f_xs
    bias2_e = np.mean(bias * bias, axis=1)
#方差
    var = re.var(axis=0, ddof=0)
    var_e = var.mean(axis=1)
#噪声
    sigma2_e = np.array([np.mean(epsilon * epsilon)] * int(p))

    return np.r_['0,2,1', bias2_e, var_e, sigma2_e, err]