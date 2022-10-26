import msprime
import itertools
import numpy as np
import sys
import multiprocessing as mp
from analytic import *

def readHapMap(path2Map):
    # assume the first row is header, so we ignore it
    bps = []
    cMs = []
    with open(path2Map) as f:
        f.readline()
        line = f.readline()
        while line:
            _, bp, _, cM = line.strip().split()
            bps.append(int(bp))
            cMs.append(float(cM))
            line = f.readline()
    return np.array(bps), np.array(cMs)


def bp2Morgan(bp, bps, cMs):
    # bps: a list of basepair position
    # cMs: a list of geneticMap position in cM corresponding to bps
    assert(len(bps) == len(cMs))
    i = np.searchsorted(bps, bp, side='left')
    if bps[i] == bp:
        return cMs[i]
    elif i == 0:
        return cMs[0]*(bp/bps[0])
    else:
        left_bp, right_bp = bps[i-1], bps[i]
        left_cM, right_cM = cMs[i-1], cMs[i]
        return left_cM + (right_cM - left_cM)*(bp - left_bp)/(right_bp - left_bp)

def ibd_segments(ts, a, b, bps, cMs, maxGen=np.inf, minLen=4):
    trees_iter = ts.trees()
    #next(trees_iter) # I don't want the first tree because recombination map doesn't cover it
    tree = next(trees_iter)
    last_mrca = tree.mrca(a, b)
    last_t = tree.time(last_mrca) if last_mrca != -1 else np.inf
    last_left = bp2Morgan(tree.interval[0], bps, cMs)
    segment_lens = []
    count = 0
    for tree in trees_iter:
        mrca = tree.mrca(a, b)
        if mrca != last_mrca:
            count += 1
            left = bp2Morgan(tree.interval[0], bps, cMs)
            #print(f'mrca changed, segment length: {left-last_left}')
            if left - last_left >= minLen and last_mrca != -1:
                segment_lens.append(left - last_left)
            last_mrca = mrca
            last_t = tree.time(mrca) if last_mrca != -1 else np.inf
            last_left = left
    # take care of the last segment
    if last_t <= maxGen and cMs[-1] - last_left >= minLen:
        segment_lens.append(cMs[-1] - last_left)
    #print(f'number of segments between {a} and {b}: {len(segment_lens)}')
    #print(f'number of times MRCA changes: {count}')
    return segment_lens


def getPath(tree, s, t):
    # return a list of nodes that specify the path from s to t
    # s is not in the list but t is
    # assume s is lower in the tree than t!
    path = []
    p = tree.parent(s)
    while p != t:
        path.append(p)
        p = tree.parent(p)
    assert(p == t)
    path.append(p)
    return path


def ibd_segments_full_ARGs(ts, a, b, bps, cMs, maxGen=np.inf, minLen=4):
    # this is used when only 1 haplotype is sampled from each island
    segment_lens = []
    trees_iter = ts.trees()
    next(trees_iter) # I don't want the first tree because recombination map doesn't cover it
    for tree in trees_iter:
        left = bp2Morgan(tree.interval[0], bps, cMs)
        right = bp2Morgan(tree.interval[1], bps, cMs)
        if right - left >= minLen and (tree.mrca(a,b) == -1 or (tree.mrca(a,b) != -1 and tree.tmrca(a,b) < maxGen)):
            segment_lens.append(right - left)
    return segment_lens

    tree = next(trees_iter)
    last_mrca = tree.mrca(a, b)
    last_pathA, last_pathB = getPath(tree, a, last_mrca), getPath(tree, a, last_mrca)
    last_left = bp2Morgan(tree.interval[0], bps, cMs)
    segment_lens = []
    for tree in trees_iter:
        mrca = tree.mrca(a, b)
        pathA, pathB = getPath(tree, a, mrca), getPath(tree, b, mrca)
        if mrca != last_mrca  or pathA != last_pathA or pathB != last_pathB:
            left = bp2Morgan(tree.interval[0], bps, cMs)
            if last_mrca <= maxGen and left - last_left >= minLen:
                segment_lens.append(left - last_left)
            last_mrca = mrca
            last_left = left
            last_pathA = pathA
            last_pathB = pathB
    # take care of the last segment
    if last_mrca <= maxGen and cMs[-1] - last_left >= minLen:
        segment_lens.append(cMs[-1] - last_left)
    return segment_lens


def simAndGetIBD_two_island(sampling, demography, recombMap, bps, cMs, maxGen=np.inf, minLen=4.0):
    ts = msprime.sim_ancestry(sampling, demography=demography, recombination_rate=recombMap, \
        record_provenance=False)
    ibd = []
    numPopA, numPopB = sampling['A'], sampling['B']
    for id1, id2 in itertools.product(range(2*numPopA), range(2*numPopA, 2*(numPopA + numPopB))):
        ibd.extend(ibd_segments(ts, id1, id2, bps, cMs, maxGen=maxGen, minLen=minLen))
    return ibd


def simAndGetIBD_two_island_fullARGs(demography, recombMap, bps, cMs, maxGen=np.inf, minLen=4.0, endTime=np.inf):
    ts = msprime.sim_ancestry({'A':1, 'B':1}, demography=demography, \
        recombination_rate=recombMap, record_provenance=False, \
            record_full_arg=True, ploidy=1, end_time=endTime)
    ibd = ibd_segments_full_ARGs(ts, 0, 1, bps, cMs, maxGen=maxGen, minLen=minLen)
    return ibd


def getCoalTimeAlongChromosome(ts, a, b, bps, cMs):
    # return a list of coalescent time along the genome of haplotype a, b
    # also a list of the length of each block
    trees_iter = ts.trees()
    next(trees_iter)
    t = []
    weight = []
    for tree in trees_iter:
        tmrca = tree.tmrca(a, b)
        left_bp, right_bp = tree.interval
        left_cM, right_cM = bp2Morgan(left_bp, bps, cMs), bp2Morgan(right_bp, bps, cMs)
        t.append(tmrca)
        weight.append(right_cM - left_cM)
    return t, weight

def multi_run(fun, prms, processes = 4, output=False):
    """Implementation of running in Parallel.
    fun: Function
    prms: The Parameter Files
    processes: How many Processes to use"""
    if output:
        print(f"Running {len(prms)} total jobs; {processes} in parallel.")
    
    if len(prms)>1:
        if output:
            print("Starting Pool of multiple workers...")    
        with mp.Pool(processes = processes) as pool:
            results = pool.starmap(fun, prms)
    elif len(prms)==1:
        if output:
            print("Running single process...")
        results = fun(*prms[0])
    else:
        raise RuntimeWarning("Nothing to run! Please check input.")
    return results

#############################################################################
#############################################################################


def ibd_segments_full_ARGs_cohort(ts, a, b, bps, cMs, maxGen=np.inf, minLen=4):
    segment_lens = []
    ts = ts.simplify([a,b], keep_unary=True)
    trees_iter = ts.trees()
    tree = next(trees_iter)
    last_mrca = tree.mrca(0, 1)
    last_pathA, last_pathB = getPath(tree, 0, last_mrca), getPath(tree, 1, last_mrca)
    last_left = bp2Morgan(tree.interval[0], bps, cMs)
    segment_lens = []
    for tree in trees_iter:
        mrca = tree.mrca(0, 1)
        pathA, pathB = getPath(tree, 0, mrca), getPath(tree, 1, mrca)
        if mrca != last_mrca  or pathA != last_pathA or pathB != last_pathB:
            left = bp2Morgan(tree.interval[0], bps, cMs)
            if last_mrca <= maxGen and left - last_left >= minLen:
                segment_lens.append(left - last_left)
            last_mrca = mrca
            last_left = left
            last_pathA = pathA
            last_pathB = pathB
    # take care of the last segment
    if last_mrca <= maxGen and cMs[-1] - last_left >= minLen:
        segment_lens.append(cMs[-1] - last_left)
    return segment_lens

def simAndGetIBD_two_island_chrom(sampling, demography, ch, end_time=500, random_seed=1, maxGen=np.inf, minLen=4.0):
    # read HapMap
    path2Map = f"/mnt/archgen/users/yilei/Data/Hapmap/genetic_map_GRCh37_chr{ch}.txt"
    recombMap = msprime.RateMap.read_hapmap(path2Map)
    bps, cMs = readHapMap(path2Map)

    ploidy = 2
    if end_time < np.inf:
        ts = msprime.sim_ancestry(sampling, demography=demography, recombination_rate=recombMap, \
            record_provenance=False, ploidy=ploidy, end_time=end_time, random_seed=random_seed)
    else:
        print(f'simulating ARG until TMRCA...')
        ts = msprime.sim_ancestry(sampling, demography=demography, recombination_rate=recombMap, \
            record_provenance=False, ploidy=ploidy, random_seed=random_seed)
    ibd = []
    numPopA, numPopB = sampling['A'], sampling['B']
    for id1, id2 in itertools.product(range(ploidy*numPopA), range(ploidy*numPopA, ploidy*(numPopA + numPopB))):
        ibd.extend(ibd_segments(ts, id1, id2, bps, cMs, maxGen=maxGen, minLen=minLen))

    # save tree seq file
    return ibd


def simAndGetIBD_two_island_chrom_fullARG(sampling, demography, ch, end_time=500, random_seed=1, maxGen=np.inf, minLen=4.0):
    # read HapMap
    path2Map = f"/mnt/archgen/users/yilei/Data/Hapmap/genetic_map_GRCh37_chr{ch}.txt"
    recombMap = msprime.RateMap.read_hapmap(path2Map)
    bps, cMs = readHapMap(path2Map)

    ploidy = 2
    if end_time < np.inf:
        ts = msprime.sim_ancestry(sampling, demography=demography, recombination_rate=recombMap, \
            record_provenance=False, ploidy=ploidy, end_time=end_time, random_seed=random_seed, record_full_arg=True)
    else:
        print(f'simulating ARG until TMRCA...')
        ts = msprime.sim_ancestry(sampling, demography=demography, recombination_rate=recombMap, \
            record_provenance=False, ploidy=ploidy, random_seed=random_seed, record_full_arg=True)
    ibd = []
    numPopA, numPopB = sampling['A'], sampling['B']
    for id1, id2 in itertools.product(range(ploidy*numPopA), range(ploidy*numPopA, ploidy*(numPopA + numPopB))):
        ibd.extend(ibd_segments_full_ARGs_cohort(ts, id1, id2, bps, cMs, maxGen=maxGen, minLen=minLen))

    # save tree seq file
    return ibd


def simAndGetIBD_two_island_ind(sampling, demography, processes, chs=range(1,23), end_time=1000, random_seed=1, maxGen=np.inf, minLen=4.0, maxLen=15.0):
    prms = [[sampling, demography, ch, end_time, random_seed, maxGen, minLen] for ch in chs]
    results = multi_run(simAndGetIBD_two_island_chrom, prms, processes=processes, output=False)
    aggregated = []
    for result in results:
        aggregated.extend(result)
    aggregated = np.array(aggregated)
    print(f'number of ibd segments: {len(aggregated)}')
    # start inference
    if len(aggregated) == 0:
        print('not enough segments to make useful inference...')
        sys.exit()
    else:
        print(f'mean of observed segments: {np.mean(aggregated)}')
        lambda_exp_mle, lambda_exp_se = two_island_noGeneFlow_constNe_truncExp(aggregated, minLen, maxLen)
        chrlens = np.array([286.279, 268.840, 223.361, 214.688, 204.089, 192.040, 187.221, 168.003, 166.359, 181.144, 158.219, 174.679, 125.706, 120.203, 141.860, 134.038, 128.491, 117.709, 107.734, 108.267, 62.786, 74.110])
        twoIsland_mle, twoIsland_se = two_island_noGeneFlow_constNe_MLE_multiStart(aggregated, minLen, maxLen, 0.1, chrlens, 2*sampling['A']*2*sampling['B'])
    return 50*lambda_exp_mle, 50*lambda_exp_se, twoIsland_mle, twoIsland_se


def simAndGetIBD_two_island_ind_fullARG(sampling, demography, processes, chs=range(1,23), end_time=1000, random_seed=1, maxGen=np.inf, minLen=5.0, maxLen=15.0, aggregated=None):
    if aggregated is None:
        prms = [[sampling, demography, ch, end_time, random_seed, maxGen, minLen] for ch in chs]
        results = multi_run(simAndGetIBD_two_island_chrom_fullARG, prms, processes=processes, output=False)
        aggregated = []
        for result in results:
            aggregated.extend(result)
        aggregated = np.array(aggregated)
    print(f'number of ibd segments: {len(aggregated)}')
    # start inference
    if len(aggregated) == 0:
        print('not enough segments to make useful inference...')
        sys.exit()
    else:
        print(f'mean of observed segments: {np.mean(aggregated)}')
        lambda_exp_mle, lambda_exp_se = two_island_noGeneFlow_constNe_truncExp(aggregated, minLen, maxLen)
        chrlens = np.array([286.279, 268.840, 223.361, 214.688, 204.089, 192.040, 187.221, 168.003, 166.359, 181.144, 158.219, 174.679, 125.706, 120.203, 141.860, 134.038, 128.491, 117.709, 107.734, 108.267, 62.786, 74.110])
        twoIsland_mle, twoIsland_se = two_island_noGeneFlow_constNe_MLE_multiStart(aggregated, minLen, maxLen, 0.1, chrlens, 2*sampling['A']*2*sampling['B'])
    return 50*lambda_exp_mle, 50*lambda_exp_se, twoIsland_mle, twoIsland_se, aggregated