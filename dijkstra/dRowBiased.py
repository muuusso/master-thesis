from itertools import islice
import multiprocessing

import numpy as np
import networkx as nx
from networkx.classes.function import path_weight

def path_to_bit_rep(p, G):
    p = [(v, p[i+1]) for i, v in enumerate(p[:-1])]
    
    bit_rep = []
    for e in G.edges():
        if (e[0], e[1]) in p:
            bit_rep.append(1)
        else:
            bit_rep.append(0)
    
    return np.array(bit_rep)


def compute_dH(p0, p1, G):
    p0 = path_to_bit_rep(p0, G)
    p1 = path_to_bit_rep(p1, G)
    
    return np.abs(p0 - p1).sum()


def compute_DdH(ns, M=100000):
    
    G = nx.grid_2d_graph(2, ns+1)

    Gdir = nx.DiGraph(G)
    for edge in Gdir.copy().edges():
        if edge != tuple(sorted(edge)):
            Gdir.remove_edge(*edge)

    source = (0, 0)
    target = (1, ns)
    
    deltaM = np.zeros(M)
    dHM = np.zeros(M)
    
    for i in range(M):
        
        for e in Gdir.edges():
            Gdir[e[0]][e[1]]['weight'] = np.random.random()
        
        p0, p1 = list(islice(nx.shortest_simple_paths(Gdir, source, target, weight="weight"), 2))
        
        deltaM[i] = path_weight(Gdir, p1, weight="weight") - path_weight(Gdir, p0, weight="weight")
        dHM[i] = compute_dH(p0, p1, Gdir)
        
    delta = (np.mean(deltaM), np.std(deltaM, ddof=1) / np.sqrt(M))
    dH = (np.mean(dHM), np.std(dHM, ddof=1) / np.sqrt(M))
    
    return (delta, dH)


ns_max = 256
ns = [(i, 100000) for i in range(1, ns_max+1)]

with multiprocessing.Pool(ns_max) as pool:
    data = pool.starmap(compute_DdH, ns)

np.save("dijkstra.npy", data) 
