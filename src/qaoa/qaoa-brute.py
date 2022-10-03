import pickle
import numpy as np
import networkx as nx
from scipy.optimize import brute, minimize
from module import MyClass

if __name__ == '__main__':
    # Loop over all instances
    for idx in range(30):
        # Path
        path = "../../data/max_cut_" + str(idx) + "/"
        # Load graph
        graph = nx.readwrite.gpickle.read_gpickle(path + "graph")
        # Create object
        obj = MyClass(graph)
        # Optimization bounds
        ranges = ((0, np.pi), (0, np.pi / 2))
        # Function to optimize
        fun = obj.optimize_qaoa
        # Brute force on 100 x 100 grid
        res = brute(fun, ranges, Ns=100, full_output=True,
                    finish=minimize, workers=6)
        # Save results
        filename = path + "qaoa_parameters_brute"
        with open(filename, 'wb') as f:
            pickle.dump(res, f)
