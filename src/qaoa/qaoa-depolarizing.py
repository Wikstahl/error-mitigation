import pickle
import numpy as np
import networkx as nx
from scipy.optimize import minimize
from module import DepolarizingChannel, MyClass

# Initialize dictionaries
data = dict()
data_with_vd = dict()
# Loop over all instances
for idx in range(30):
    # Path
    path = "../../data/max_cut_" + str(idx) + "/"
    # Load graph
    graph = nx.readwrite.gpickle.read_gpickle(path + "graph")
    # Create object
    obj = MyClass(graph)
    # Load optimal angles for no noise
    res = pickle.load(open(path + "qaoa_parameters_brute", "rb"))
    # Optimal parameters with no noise
    x0 = res[0]
    x0_with_vd = res[0]
    # Loop over noise levels
    noise_levels = np.linspace(0, 0.1, 21)
    for key, p in enumerate(noise_levels):
        if key == 0:
            data[str(key)] = (p, res)
            data_with_vd[str(key)] = (p, res)
            continue
        # Args to minimize
        args = (DepolarizingChannel(p=p))
        # Find the minimum starting from the previous minimum
        res = minimize(obj.optimize_qaoa, x0, args=args)
        res_with_vd = minimize(obj.optimize_qaoa_with_vd, x0_with_vd, args=args)
        # Update the initial guess
        x0 = res.x
        x0_with_vd = res_with_vd.x
        # Save result to dict
        data[str(key)] = (p, res)
        data_with_vd[str(key)] = (p, res_with_vd)

    # Save results
    filename = path + "qaoa_parameters_minimize_depolarizing"
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

    filename = path + "qaoa_parameters_minimize_depolarizing_with_vd"
    with open(filename, 'wb') as f:
        pickle.dump(data_with_vd, f)
