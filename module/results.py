import pickle
import networkx
from module import MyClass, DephasingChannel, DepolarizingChannel

__all__ = ["Results"]


class Results(object):

    def __init__(self, path: str) -> None:
        # Path to data folder
        self.path = path
        # Load graph
        self.graph = networkx.readwrite.gpickle.read_gpickle(path + "graph")
        # Create object
        self.obj = MyClass(self.graph)
        # Get optimal cost
        self.min_cost = min(self.obj.cost)

    def DephasingNoise(self):
        # Load the results for dephasing noise
        res = pickle.load(
            open(self.path + "qaoa_parameters_minimize_dephasing", "rb"))
        # Number of data points
        num_dat = len(res)
        # Mitigated approximation ratio
        approxr = []
        # Loop over all data points
        for idx in range(num_dat):
            if idx == 0:
                cost = res[str(idx)][1][1]
            else:
                cost = res[str(idx)][1].fun
            # Calculate the mitigated approximation ratio
            approxr.append(cost / self.min_cost)
        return approxr

    def DephasingNoiseWithVD(self):
        # Load the results for dephasing noise
        res = pickle.load(
            open(self.path + "qaoa_parameters_minimize_dephasing_with_vd", "rb"))
        # Number of data points
        num_dat = len(res)
        # Mitigated approximation ratio
        mitigated_approxr = []
        # Loop over all data points
        for idx in range(num_dat):
            if idx == 0:
                cost = res[str(idx)][1][1]
            else:
                cost = res[str(idx)][1].fun
            # Calculate the mitigated approximation ratio
            mitigated_approxr.append(cost / self.min_cost)
        return mitigated_approxr

    def DepolarizingNoise(self):
        # Load the results for dephasing noise
        res = pickle.load(
            open(self.path + "qaoa_parameters_minimize_depolarizing", "rb"))
        # Number of data points
        num_dat = len(res)
        # Mitigated approximation ratio
        approxr = []
        # Loop over all data points
        for idx in range(num_dat):
            if idx == 0:
                cost = res[str(idx)][1][1]
            else:
                cost = res[str(idx)][1].fun
            # Calculate the mitigated approximation ratio
            approxr.append(cost / self.min_cost)
        return approxr

    def DepolarizingNoiseWithVD(self):
        # Load the results for dephasing noise
        res = pickle.load(
            open(self.path + "qaoa_parameters_minimize_depolarizing_with_vd", "rb"))
        # Number of data points
        num_dat = len(res)
        # Mitigated approximation ratio
        mitigated_approxr = []
        # Loop over all data points
        for idx in range(num_dat):
            if idx == 0:
                cost = res[str(idx)][1][1]
            else:
                cost = res[str(idx)][1].fun
            # Calculate the mitigated approximation ratio
            mitigated_approxr.append(cost / self.min_cost)
        return mitigated_approxr

    def get_angles(self):
        # Load the results for depolarizing noise
        res_dep = pickle.load(
            open(self.path + "qaoa_parameters_minimize_depolarizing", "rb"))
        # Load the results for dephasing noise
        res_z = pickle.load(
            open(self.path + "qaoa_parameters_minimize_dephasing", "rb"))

        # Number of data points
        num_dat = len(res_dep)

        alpha_dep = []
        beta_dep = []
        alpha_z = []
        beta_z = []
        for idx in range(num_dat):
            if idx == 0:
                alpha_dep.append(res_dep[str(idx)][1][0][0])
                beta_dep.append(res_dep[str(idx)][1][0][1])
                alpha_z.append(res_z[str(idx)][1][0][0])
                beta_z.append(res_z[str(idx)][1][0][1])
            else:
                alpha_dep.append(res_dep[str(idx)][1].x[0])
                beta_dep.append(res_dep[str(idx)][1].x[1])
                alpha_z.append(res_z[str(idx)][1].x[0])
                beta_z.append(res_z[str(idx)][1].x[1])

        return alpha_dep, beta_dep, alpha_z, beta_z

    def get_vd_angles(self):
        # Load the results for depolarizing noise
        res_dep = pickle.load(
            open(self.path + "qaoa_parameters_minimize_depolarizing_with_vd", "rb"))
        # Load the results for dephasing noise
        res_z = pickle.load(
            open(self.path + "qaoa_parameters_minimize_dephasing_with_vd", "rb"))

        # Number of data points
        num_dat = len(res_dep)

        alpha_dep = []
        beta_dep = []
        alpha_z = []
        beta_z = []
        for idx in range(num_dat):
            if idx == 0:
                alpha_dep.append(res_dep[str(idx)][1][0][0])
                beta_dep.append(res_dep[str(idx)][1][0][1])
                alpha_z.append(res_z[str(idx)][1][0][0])
                beta_z.append(res_z[str(idx)][1][0][1])
            else:
                alpha_dep.append(res_dep[str(idx)][1].x[0])
                beta_dep.append(res_dep[str(idx)][1].x[1])
                alpha_z.append(res_z[str(idx)][1].x[0])
                beta_z.append(res_z[str(idx)][1].x[1])

        return alpha_dep, beta_dep, alpha_z, beta_z
