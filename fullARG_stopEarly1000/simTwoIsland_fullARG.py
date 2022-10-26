import numpy as np
import msprime
import argparse
import sys
import matplotlib.pyplot as plt
import pickle



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate Two-Island model and plot IBD distribution')
    parser.add_argument('-N', action="store", dest="Ne", type=float, required=True,
                        help="ancestral population size")
    parser.add_argument('-T', action="store", dest="T", type=float, required=True,
                        help="split time in generations.")
    parser.add_argument('-p', action="store", dest="processes", type=int, required=False, default=12,
                        help="Number of processes to use.")
    args = parser.parse_args()
    
    sys.path.insert(0, "/mnt/archgen/users/yilei/IBD/two_island_final/code")
    from analytic import *
    from ts_utility import *

    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=args.Ne)
    demography.add_population(name="B", initial_size=args.Ne)
    demography.add_population(name="AB", initial_size=2*args.Ne)
    demography.add_population_split(time=args.T, derived=["A", "B"], ancestral="AB")
    print(f'simulating two-island model with ancestral diploid pop size {args.Ne} with split time {args.T}...')

    path2Map = "/mnt/archgen/users/yilei/Data/Hapmap/genetic_map_GRCh37_chr5.txt"
    recombMap = msprime.RateMap.read_hapmap(path2Map)
    bps, cMs = readHapMap(path2Map)

    numReps = 50000
    endTime = 1000 # end simulation at 1000 generations back in time
    prms = [[demography, recombMap, bps, cMs, np.inf, 4.0, endTime] for i in range(numReps)]
    results = multi_run(simAndGetIBD_two_island_fullARGs, prms, processes=args.processes, output=True)
    aggregated = []
    dummy = [aggregated.extend(batch) for batch in results]
    f = open(f'./fullARG_stopEarly1000/pickle/two_island_T{int(args.T)}_N{int(args.Ne)}_fullARG', 'wb')
    pickle.dump(aggregated, f)
    print(f'total number of IBD segments found: {len(aggregated)}')

    minl, maxl = 4, 20
    step = 0.1
    bins = np.arange(minl, maxl+step, step)
    midpoint = (bins[:-1] + bins[1:])/2
    meanNumIBD_eq5 = [two_island_noGeneFlow_constNe_eq5(cMs[-1]/100, l/100, args.T, args.Ne) for l in midpoint]
    meanNumIBD_eq4 = [two_island_noGeneFlow_constNe_eq4(cMs[-1]/100, l/100, args.T, args.Ne) for l in midpoint]
    # each bin is 0.1cM, so we multiply by 0.001M here
    meanNumIBD_eq5 = 0.001*numReps*np.array(meanNumIBD_eq5)
    meanNumIBD_eq4 = 0.001*numReps*np.array(meanNumIBD_eq4)
    plt.hist(aggregated, bins=bins)
    plt.plot(midpoint, meanNumIBD_eq5, color='red', label='eq5')
    plt.plot(midpoint, meanNumIBD_eq4, color='black', label='eq4')
    rate = args.T/50
    normalizing_const = np.exp(-rate*minl) - np.exp(-rate*maxl)
    plt.plot(midpoint, len(aggregated)*step*rate*np.exp(-rate*midpoint)/normalizing_const, color='orange', label='exp fit')
    plt.legend(loc='upper right', fontsize='small')
    plt.title(f'T:{int(args.T)}, N_ancestral: {int(args.Ne)}')
    plt.xlabel('length of IBD in cM')
    plt.ylabel('number of IBD segments in each bin of 0.1cM')
    plt.yscale('log')
    plt.savefig(f'./fullARG_stopEarly1000/fig/two_island_T{int(args.T)}_N{int(args.Ne)}_fullARG.log.png', dpi=300)


