import pickle
import numpy as np
import networkx as nx
from module import DephasingChannel, MyClass

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
    for key, p in enumerate(np.linspace(0, .5, 26)):
        # Simulate virtual distillation
        rho_out = obj.simulate_virtual_distillation(
            rho_in,
            with_noise=DephasingChannel(p=p)
        )
        # Calculate mitigated variance
        mv = obj.mitigated_variance(rho_out)
        data[str(key)] = tuple([p, mv])
    # Save results
    filename = path + "thermal_variance_dephasing"
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
