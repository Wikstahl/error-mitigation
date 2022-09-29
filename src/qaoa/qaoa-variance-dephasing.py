import pickle
import numpy as np
import networkx as nx
from module import DephasingChannel, MyClass

# Initialize dictionary
data = dict()
# Loop over all instances
for idx in range(30):
    # Path
    path = "../../data/max_cut_"+str(idx)+"/"
    # Load graph
    graph = nx.readwrite.gpickle.read_gpickle(path + "graph")
    # Load QAOA parameters
    params = pickle.load(open(path+"qaoa_parameters_minimize_dephasing", "rb"))
    # Load QAOA with VD parameters
    params_with_vd = pickle.load(
        open(path+"qaoa_parameters_minimize_dephasing_with_vd", "rb"))
    # Create object
    obj = MyClass(graph)
    # Loop over noise levels
    for key, p in enumerate(np.linspace(0, 0.1, 21)):
        if p == 0:
            # Get the optimal parameters
            alpha, beta = params[str(key)][1][0]
            # Get the optimal parameters with VD
            alpha_vd, beta_vd = params_with_vd[str(key)][1][0]
            # Simulate QAOA
            rho = obj.simulate_qaoa((alpha, beta))
            # Simulate QAOA with VD parameters
            rho_vd = obj.simulate_qaoa((alpha_vd, beta_vd))
            # Simulate Virtual Distillation
            rho_out = obj.simulate_virtual_distillation(rho_vd)
            # Calculate the variance of the unmitigated estimator
            var = obj.unmitigated_variance(rho)
            # Calculate the variance of the mitigated estimator
            var_m = obj.mitigated_variance(rho_out)
        else:
            # Get the optimal parameters
            alpha, beta = params[str(key)][1].x
            # Get the optimal parameters with VD
            alpha_vd, beta_vd = params_with_vd[str(key)][1].x
            # Noise type
            noise = DephasingChannel(p=p)
            # Simulate QAOA
            rho = obj.simulate_qaoa((alpha, beta), with_noise=noise)
            # Simulate QAOA with VD parameters
            rho_vd = obj.simulate_qaoa((alpha_vd, beta_vd), with_noise=noise)
            # Simulate Virtual Distillation
            rho_out = obj.simulate_virtual_distillation(
                rho_vd, with_noise=noise)
            # Calculate the variance of the unmitigated estimator
            var = obj.unmitigated_variance(rho)
            # Calculate the variance of the mitigated estimator
            var_m = obj.mitigated_variance(rho_out)
        # Save data to dict
        data[str(key)] = (p, var, var_m)
    # Save results
    filename = path + "qaoa_variance_equal_error_dephasing"
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
