import pickle
import numpy as np
import networkx as nx
from module import DephasingChannel, DepolarizingChannel, MyClass, Results
from module import drift

# Loop over all instances
for idx in range(30):
    # Path
    path = "../../data/max_cut_" + str(idx) + "/"
    # Load graph
    graph = nx.readwrite.gpickle.read_gpickle(path + "graph")
    # Create object
    obj = MyClass(graph)
    # Create result object
    res = Results(path)
    # Get optimal angles
    alpha_dep, beta_dep, alpha_z, beta_z = res.get_angles()
    alpha_with_vd_dep, beta_with_vd_dep, alpha_with_vd_z, beta_with_vd_z = res.get_vd_angles()
    # Get the noiseless angles
    alpha, beta = alpha_dep[0], beta_dep[0]
    # Simulate QAOA
    psi = obj.simulate_qaoa((alpha, beta))
    # Loop over noise levels
    noise_levels = np.linspace(0, 0.1, 21)
    # Initialize empty arrays
    drift_z = []
    drift_with_vd_z = []
    drift_dep = []
    drift_with_vd_dep = []
    for key, p in enumerate(noise_levels):
        # Simulate QAOA with noise
        rho_z = obj.simulate_qaoa(
            (alpha_z[key], beta_z[key]), with_noise=DephasingChannel(p=p))
        rho_with_vd_z = obj.simulate_qaoa(
            (alpha_with_vd_z[key], beta_with_vd_z[key]), with_noise=DephasingChannel(p=p))
        rho_dep = obj.simulate_qaoa(
            (alpha_dep[key], beta_dep[key]), with_noise=DepolarizingChannel(p=p))
        rho_with_vd_dep = obj.simulate_qaoa(
            (alpha_with_vd_dep[key], beta_with_vd_dep[key]), with_noise=DepolarizingChannel(p=p))
        drift_z.append(drift(psi, rho_z))
        drift_dep.append(drift(psi, rho_dep))
        drift_with_vd_z.append(drift(psi, rho_with_vd_z))
        drift_with_vd_dep.append(drift(psi, rho_with_vd_dep))

    data = np.array([drift_z, drift_dep])
    data_with_vd = np.array([drift_with_vd_z, drift_with_vd_dep])
    # Save results
    filename = path + "qaoa_drift"
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

    # Save results
    filename = path + "qaoa_drift_with_vd"
    with open(filename, 'wb') as f:
        pickle.dump(data_with_vd, f)
