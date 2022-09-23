import pickle
import numpy as np
import networkx as nx
from scipy.optimize import minimize
from module import DephasingChannel, MyClass

# Initialize dictionary
data = dict()
# Loop over all instances
for idx in range(30):
    # Path
    path = "../../data/max_cut_" + str(idx) + "/"
    # Load graph
    graph = nx.readwrite.gpickle.read_gpickle(path + "graph")
    # Create object
    obj = MyClass(graph)
    # Load optimal angles
    res = pickle.load(open(path + "qaoa_parameters_brute_with_vd", "rb"))
    # Optimal parameters with no noise
    x0 = res[0]
    # Loop over noise levels
    noise_levels = np.linspace(0, 0.1, 21)
    for key, p in enumerate(noise_levels):
        if key == 0:
            data[str(key)] = (p, res)
            continue
        # Args to minimize
        args = (DephasingChannel(p=p))
        # Find the minimum starting from the previous minimum
        res = minimize(obj.optimize_qaoa_with_vd, x0, args=args)
        # Update the initial guess
        x0 = res.x
        # Save result to dict
        data[str(key)] = (p, res)
    # Save results
    filename = path + "qaoa_parameters_minimize_dephasing_with_vd"
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
