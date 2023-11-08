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

    def noisyqaoa_perfectvd(self):
        # Load the results for dephasing noise
        res = pickle.load(
            open(self.path + "qaoa_parameters_minimize_dephasing", "rb"))
        
        # Number of data points
        num_dat = len(res)
        # Mitigated approximation ratio
        approxr = []

        # Loop over all data points
        for idx in range(num_dat):
            error_p = res[str(idx)][0] # error probability
            opt_res = res[str(idx)][1] # optimization results
            if idx == 0:
                params = opt_res[0]
            else:
                params = tuple(opt_res.x)
            rho = self.obj.simulate_qaoa(params, DephasingChannel(error_p))
            
            # perform virtual distillation
        return approxr

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

        res = pickle.load(open(self.path + "qaoa_variance_dephasing", "rb"))
        var = [res[str(i)][1] for i in range(num_dat)]
        return approxr, var

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

        res = pickle.load(
            open(self.path + "qaoa_variance_dephasing_with_vd", "rb"))
        m_var = [res[str(i)][1] for i in range(num_dat)]

        return mitigated_approxr, m_var

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

        res = pickle.load(open(self.path + "qaoa_variance_depolarizing", "rb"))
        var = [res[str(i)][1] for i in range(num_dat)]
        return approxr, var

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

        res = pickle.load(
            open(self.path + "qaoa_variance_depolarizing_with_vd", "rb"))
        m_var = [res[str(i)][1] for i in range(num_dat)]
        return mitigated_approxr, m_var

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
