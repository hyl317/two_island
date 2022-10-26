import numpy as np
import msprime
import argparse
import sys
import matplotlib.pyplot as plt
import pickle



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate Two-Island model and plot IBD distribution')
    parser.add_argument('-N', action="store", dest="Ne", type=int, required=True,
                        help="ancestral population size")
    parser.add_argument('-T', action="store", dest="T", type=int, required=True,
                        help="split time in generations.")
    parser.add_argument('-r', action="store", dest="r", type=int, required=True,
                        help="Number of replicates.")
    parser.add_argument('-e', action="store", dest="end", type=int, required=False, default=1000,
                        help="simulation end time")
    parser.add_argument('-p', action="store", dest="processes", type=int, required=False, default=12,
                        help="Number of processes to use.")
    args = parser.parse_args()
    
    sys.path.insert(0, "/mnt/archgen/users/yilei/IBD/two_island_final/code")
    from analytic import *
    from ts_utility import *

    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=args.Ne/2)
    demography.add_population(name="B", initial_size=args.Ne/2)
    demography.add_population(name="AB", initial_size=args.Ne)
    demography.add_population_split(time=args.T, derived=["A", "B"], ancestral="AB")
    print(f'simulating two-island model with ancestral pop size {args.Ne} with split time {args.T}...')

    
    sampling = {'A':10, 'B':10}
    with open(f'./results/T{args.T}_N{args.Ne}_end{args.end}_notFullARG.results', 'w') as out:
        out.write(f'#replicate\texp_mle\texp_se\t2island_T_mle\t2island_T_se\t2island_N_mle\t2island_N_se\n')
        for r in range(1, 1+args.r):
            exp_mle, exp_se, twoIsland_mle, twoIsland_se = \
                simAndGetIBD_two_island_ind(sampling, demography, \
                    args.processes, end_time=args.end, random_seed=r, \
                        maxGen=np.inf, minLen=4.0, maxLen=15.0)
            out.write(f'batch{r}\t{round(exp_mle,3)}\t{round(exp_se,3)}\t{round(twoIsland_mle[0],3)}\t{round(twoIsland_se[0],3)}\t{round(twoIsland_mle[1],3)}\t{round(twoIsland_se[1],3)}\n')    
