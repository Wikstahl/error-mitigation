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
            open(self.path + "qaoa_parameters_brute_dephasing", "rb"))
        # Number of data points
        num_dat = len(res)
        # Store the optimal alpha angles for each data point in a list
        alpha = [res[str(i)][1][0][0] for i in range(num_dat)]
        # Store the optimal beta angles for each data point in a list
        beta = [res[str(i)][1][0][1] for i in range(num_dat)]
        # Store the unmitigated approximation ratio in a list
        approxr = [res[str(i)][1][1] / self.min_cost for i in range(num_dat)]
        # Store all error-data points in a list
        x = [res[str(i)][0] for i in range(num_dat)]

        res = pickle.load(open(self.path + "qaoa_variance_dephasing","rb"))
        var = [res[str(i)][1] for i in range(num_dat)]
        m_var = [res[str(i)][2] for i in range(num_dat)]

        # Simulate QAOA with dephasing noise to get the density matrix output
        m_approxr = []  # Init empty list
        for i in range(num_dat):
            if x[i] == 0:
                rho = self.obj.simulate_qaoa(
                    (alpha[i], beta[i]), with_noise=None)
            else:
                rho = self.obj.simulate_qaoa(
                    (alpha[i], beta[i]), with_noise=DephasingChannel(p=x[i]))
            # Calculate the mitigated cost given rho
            m_cost = self.obj.mitigated_cost(rho)
            # Calculate the mitigated approximation ratio
            m_approxr.append(m_cost / self.min_cost)

        return approxr, m_approxr, var, m_var


    def DepolarizingNoise(self):
        # Load the results for depolarizing noise
        res = pickle.load(
            open(self.path + "qaoa_parameters_brute_depolarizing", "rb"))

        # Number of data points
        num_dat = len(res)
        # Store the optimal alpha angles for each data point in a list
        alpha = [res[str(i)][1][0][0] for i in range(num_dat)]
        # Store the optimal beta angles for each data point in a list
        beta = [res[str(i)][1][0][1] for i in range(num_dat)]
        # Store the unmitigated approximation ratio in a list
        approxr = [res[str(i)][1][1] / self.min_cost for i in range(num_dat)]
        # Store all error-data points in a list
        x = [res[str(i)][0] for i in range(num_dat)]

        res = pickle.load(open(self.path + "qaoa_variance_depolarizing","rb"))
        var = [res[str(i)][1] for i in range(num_dat)]
        m_var = [res[str(i)][2] for i in range(num_dat)]

        # Simulate QAOA with depolarizing noise
        m_approxr = []
        for i in range(num_dat):
            if x[i] == 0:
                rho = self.obj.simulate_qaoa((alpha[i], beta[i]), with_noise=None)
            else:
                rho = self.obj.simulate_qaoa(
                    (alpha[i], beta[i]), with_noise=DepolarizingChannel(p=x[i]))
            # Calculate the mitigated cost
            m_cost = self.obj.mitigated_cost(rho, p=x[i])
            # Calculate the mitigated approximation ratio
            m_approxr.append(m_cost / self.min_cost)

        return approxr, m_approxr, var, m_var

    def get_angles(self):
        # Load the results for depolarizing noise
        res_dep = pickle.load(
            open(self.path + "qaoa_parameters_brute_depolarizing", "rb"))
        # Load the results for dephasing noise
        res_z = pickle.load(
            open(self.path + "qaoa_parameters_brute_dephasing", "rb"))

        # Number of data points
        num_dat = len(res_dep)

        # Store the optimal alpha angles for each data point in a list
        alpha_dep = [res_dep[str(i)][1][0][0] for i in range(num_dat)]
        # Store the optimal beta angles for each data point in a list
        beta_dep = [res_dep[str(i)][1][0][1] for i in range(num_dat)]

        # Store the optimal alpha angles for each data point in a list
        alpha_z = [res_z[str(i)][1][0][0] for i in range(num_dat)]
        # Store the optimal beta angles for each data point in a list
        beta_z = [res_z[str(i)][1][0][1] for i in range(num_dat)]

        return alpha_dep, beta_dep, alpha_z, beta_z
