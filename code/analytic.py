import math
import numpy as np
import numdifftools as ndt
from scipy.optimize import minimize
from scipy.optimize import newton


def two_island_noGeneFlow_constNe_eq5(G, L, T, N):
    # calculate expected number of IBD segments of length L
    # for a chromosome of length G,
    # in a two island model with split time T (in generations)
    # both L, G should be in Morgan
    # don't account for chromosome edge effect, i.e, using eq5 from Ringbauer et.al, 2017
    tmp = 1 + 4*N*L
    poly = T**2/tmp + 4*T*N/tmp**2 + 8*N**2/tmp**3
    exp = 4*G*np.exp(-2*L*T)
    return exp*poly

def two_island_noGeneFlow_constNe_eq4(G, L, T, N):
    # same as two_island_noGeneFlow_constNe_eq5 but with chromosome edge effect
    # i.e, using eq4 from Ringbauer et.al, 2017
    tmp = 1 + 4*N*L
    part1 = (4*np.exp(-2*L*T)/tmp)*(T + 2*N/tmp)
    part2 = 4*(G-L)*np.exp(-2*L*T)*(T**2/tmp + 4*T*N/tmp**2 + 8*N**2/tmp**3)
    return part1 + part2

def two_island_noGeneFlow_constNe_negliklihood(params, histogram, binMidpoint, func, G, numPairs):
    # params = [split_time, ancestral_population_size]
    # G: chromosome length, could be a list of values or a single numeric
    # func: which function to use to calculate the expected sharing of segments of certain length
    assert(len(histogram) == len(binMidpoint))
    step = binMidpoint[1] - binMidpoint[0]
    if isinstance(G, float):
        lambdas = func(G, binMidpoint, params[0], params[1])*step
    else:
        lambdas = np.zeros((len(G), len(binMidpoint)))
        for i, g in enumerate(G):
            lambdas[i] += func(g, binMidpoint, params[0], params[1])
    lambdas = np.sum(lambdas, axis=0)*numPairs*step
    loglik_each_bin = histogram*np.log(lambdas) - lambdas
    return -np.sum(loglik_each_bin)

def two_island_noGeneFlow_constNe_MLE(ibd_segments, minLen, maxLen, step, G, numPairs):
    # use only ibd segments between minLen and maxLen
    # bin all segments at bin size equal to step
    # all values in unit centiMorgan
    bins = np.arange(minLen, maxLen + step, step)
    histogram = np.zeros(len(bins)-1)
    for i in range(len(bins)-1):
        low, high = bins[i], bins[i+1]
        histogram[i] = np.sum(np.logical_and(ibd_segments >= low, ibd_segments < high))
    binMidpoint = (bins[:-1] + bins[1:])/2.0
    #histogram = histogram/(500*10*10)
    kargs = (histogram, binMidpoint/100, two_island_noGeneFlow_constNe_eq5, G/100, numPairs)
    res = minimize(two_island_noGeneFlow_constNe_negliklihood, [10, 2000], args=kargs, method='L-BFGS-B', bounds=[(0, np.inf), (0, np.inf)])

    # estimate confidence interval
    Hfun = ndt.Hessian(two_island_noGeneFlow_constNe_negliklihood, step=1e-4, full_output=True)
    h, info = Hfun(res.x, *kargs)
    se = np.sqrt(np.diag(np.linalg.inv(h)))
    print('###########################################################')
    print(f'split time: {round(res.x[0],2)}({round(res.x[0]-1.96*se[0],2)} - {round(res.x[0]+1.96*se[0],2)})')
    print(f'ancestral pop size: {round(res.x[1],2)}({round(res.x[1]-1.96*se[1],2)} - {round(res.x[1]+1.96*se[1],2)})')
    print(res)
    print('###########################################################')
    return res.x, se

def two_island_noGeneFlow_constNe_MLE_multiStart(ibd_segments, minLen, maxLen, step, G, numPairs):
    # use only ibd segments between minLen and maxLen
    # bin all segments at bin size equal to step
    # all values in unit centiMorgan
    # run BFGS from multiple starting point
    bins = np.arange(minLen, maxLen + step, step)
    histogram = np.zeros(len(bins)-1)
    for i in range(len(bins)-1):
        low, high = bins[i], bins[i+1]
        histogram[i] = np.sum(np.logical_and(ibd_segments >= low, ibd_segments < high))
    binMidpoint = (bins[:-1] + bins[1:])/2.0
    kargs = (histogram, binMidpoint/100, two_island_noGeneFlow_constNe_eq5, G/100, numPairs)
    starts = [[25, 100], [25, 2000], [25, 10000]]
    curr_min = np.inf
    res = None
    for start in starts:
        res_tmp = minimize(two_island_noGeneFlow_constNe_negliklihood, start, args=kargs, method='L-BFGS-B', bounds=[(0, np.inf), (0, np.inf)])
        if res_tmp.fun < curr_min:
            curr_min = res_tmp.fun
            res = res_tmp

    # estimate confidence interval
    Hfun = ndt.Hessian(two_island_noGeneFlow_constNe_negliklihood, step=1e-4, full_output=True)
    h, info = Hfun(res.x, *kargs)
    se = np.sqrt(np.diag(np.linalg.inv(h)))
    print(f'hessian info: {info}')
    print(f'inverse of hessian: {np.linalg.inv(h)@h}')
    print('###########################################################')
    print(f'split time: {round(res.x[0],2)}({round(res.x[0]-1.96*se[0],2)} - {round(res.x[0]+1.96*se[0],2)})')
    print(f'ancestral pop size: {round(res.x[1],2)}({round(res.x[1]-1.96*se[1],2)} - {round(res.x[1]+1.96*se[1],2)})')
    print(f'curr_min: {curr_min}')
    print(res)
    print('###########################################################')
    return res.x, se

def two_island_noGeneFlow_constNe_truncExp(ibd_segments, minLen, maxLen):
    i = np.logical_and(ibd_segments > minLen, ibd_segments < maxLen)
    ibd_segments = ibd_segments[i]
    x_bar = np.mean(ibd_segments) - minLen
    x_0 = maxLen - minLen
    findroot = lambda x, x_bar, x_0 : 1/x - x_0*math.exp(-x*x_0)/(1 - math.exp(-x*x_0)) - x_bar
    x  = 1/x_bar # initial guess (as if there is no right truncation)
    lambda_hat = newton(findroot, x, args=(x_bar, x_0))
    fisher_info = 1/lambda_hat**2 - x_0**2/(4*math.sinh(0.5*lambda_hat*x_0)**2)
    se = 1/math.sqrt(len(ibd_segments)*fisher_info)
    print(f'split time from exp fit: {round(50*lambda_hat,3)}({round(50*(lambda_hat - 1.96*se),3)} - {round(50*(lambda_hat + 1.96*se),3)})') 
    return lambda_hat, se