import pickle
import numpy as np
import networkx as nx
from scipy.optimize import brute, fmin
from module import DepolarizingChannel, MyClass

# Initialize dictionary
data = dict()
# Loop over all instances
for idx in range(30):
    # Path
    path = "../../data/max_cut_"+str(idx)+"/"
    # Load graph
    graph = nx.readwrite.gpickle.read_gpickle(path + "graph")
    # Create object
    obj = MyClass(graph)
    # Loop over noise levels
    for key, p in enumerate(np.linspace(0,0.1,21)):
        # Args to brute
        args = ([DepolarizingChannel(p=p)])
        # Optimization bounds
        ranges = ((0, np.pi), (0, np.pi/2))
        # Brute force on 100 x 100 grid
        if p == 0:
            res = brute(obj.optimize_qaoa_with_vd, ranges, args=None, Ns=100,
                        full_output=True, finish=fmin, workers=6)
        else:
            res = brute(obj.optimize_qaoa_with_vd, ranges, args=args, Ns=100,
                        full_output=True, finish=fmin, workers=6)
        data[str(key)] = (p, res)
    # Save results
    filename = path + "qaoa_parameters_brute_depolarizing_with_vd"
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
