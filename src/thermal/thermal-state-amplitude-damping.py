import pickle
import numpy as np
import networkx as nx
from module import AmplitudeDampingChannel, MyClass

# Initialize dictionary
data = dict()
# Loop over all graphs
for idx in range(30):
    # Path
    path = "../../data/max_cut_" + str(idx) + "/"
    # Load graph
    graph = nx.readwrite.gpickle.read_gpickle(path + "graph")
    # Create object
    obj = MyClass(graph)
    # Create mixed state
    rho_in = np.diag(obj.mixed_state())
    # Loop over noise levels
    for key, p in enumerate(np.linspace(0, .25, 13)):
        mitigated_cost = obj.mitigated_cost_explicit(rho=rho_in, with_noise=AmplitudeDampingChannel(p=p))
        data[str(key)] = tuple([p, mitigated_cost])
    # Save results
    filename = path + "thermal_equal_error_amplitude_damping"
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
