import unittest
import random
import networkx
import numpy
import cirq
from module import MyClass, DepolarizingChannel, DephasingChannel, AmplitudeDampingChannel, drift, fidelity, average_gate_fidelity


class TestGraph(unittest.TestCase):
    def setUp(self) -> None:
        # Create a random graph
        graph = networkx.generators.random_graphs.erdos_renyi_graph(n=5, p=0.5)
        # Create object
        self.obj = MyClass(graph)
        self.num_nodes = len(graph.nodes)
        # Prepare a mixed state
        self.input = numpy.diag(self.obj.mixed_state())
        # Try 3 different noise levels
        self.p = [0, .1, .2]
        # Symmeterized version of the MaxCut Hamiltonian
        self.C_2 = 0
        self.qubits = cirq.LineQubit.range(2 * self.num_nodes + 1)
        for (u, v) in graph.edges:
            self.C_2 += -1 / 4 * (1 - cirq.PauliString(cirq.Z(self.qubits[u + 1])) * cirq.PauliString(cirq.Z(self.qubits[v + 1]))) \
                - 1 / 4 * (1 - cirq.PauliString(cirq.Z(self.qubits[u + (self.num_nodes + 1)])) * cirq.PauliString(
                    cirq.Z(self.qubits[v + (self.num_nodes + 1)])))
        # Ancilla Observable
        self.X_0 = cirq.PauliString(cirq.X(self.qubits[0]))
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_mitigate_cost_explicit(self):
        for p in self.p:
            with self.subTest(p=p):
                mitigated_cost_explicit = self.obj.mitigated_cost_explicit(rho=self.input, with_noise=DepolarizingChannel(p=p))
                mitigated_cost = self.obj.mitigated_cost(rho=self.input, p=p)
                self.assertAlmostEqual(mitigated_cost_explicit, mitigated_cost, places=6)

    def test_mitigated_cost_depolarizing(self):
        for p in self.p:
            with self.subTest(p=p):
                # Run virtual distillation
                rho_out = self.obj.simulate_virtual_distillation(
                    self.input,
                    with_noise=DepolarizingChannel(p=p)
                )
                y = self.X_0 * self.C_2  # observable numerator
                numerator = y.expectation_from_density_matrix(
                    state=rho_out,
                    qubit_map={q: i for i, q in enumerate(self.qubits)}
                ).real
                denominator = self.X_0.expectation_from_density_matrix(
                    state=rho_out,
                    qubit_map={q: i for i, q in enumerate(self.qubits)}
                ).real
                mitigated_cost = (numerator / denominator)
                predicted_mitigated_cost = self.obj.mitigated_cost(
                    self.input, p=p)
                self.assertAlmostEqual(
                    mitigated_cost, predicted_mitigated_cost, places=6)

    def test_mitigated_cost_dephasing(self):
        for p in self.p:
            with self.subTest(p=p):
                # Run virtual distillation
                rho_out = self.obj.simulate_virtual_distillation(
                    self.input,
                    with_noise=DephasingChannel(p=p)
                )
                y = self.X_0 * self.C_2  # observable numerator
                numerator = y.expectation_from_density_matrix(
                    state=rho_out,
                    qubit_map={q: i for i, q in enumerate(self.qubits)}
                ).real
                denominator = self.X_0.expectation_from_density_matrix(
                    state=rho_out,
                    qubit_map={q: i for i, q in enumerate(self.qubits)}
                ).real
                mitigated_cost = (numerator / denominator)
                predicted_mitigated_cost = self.obj.mitigated_cost(
                    self.input, p=0)
                self.assertAlmostEqual(
                    mitigated_cost, predicted_mitigated_cost, places=6)

    def test_mitigated_variance_dephasing(self):
        for p in self.p:
            with self.subTest(p=p):
                var_predicted = 1 / (1 - 2*p)**(2 * self.num_nodes) \
                    * self.obj.ideal_mitigated_variance(self.input)
                # Run virtual distillation
                rho_out = self.obj.simulate_virtual_distillation(
                    self.input,
                    with_noise=DephasingChannel(p=p)
                )
                var_estimated = self.obj.mitigated_variance(rho_out)
                self.assertAlmostEqual(
                    round(var_predicted), round(var_estimated), places=6)

    def test_zero_noise_variance(self):
        """
        For zero noise the unmitigated variance should be half the mitigated
        one for a pure state
        """
        alpha = numpy.random.rand() * numpy.pi
        beta = numpy.random.rand() * numpy.pi / 2
        rho = self.obj.simulate_qaoa(params=(alpha, beta))
        var_unmitigated = self.obj.unmitigated_variance(rho)
        rho_out = self.obj.simulate_virtual_distillation(rho)
        var_mitigated = self.obj.mitigated_variance(rho_out)
        self.assertAlmostEqual(round(var_mitigated),
                               round(var_unmitigated / 2))


class TestAverageGateFidelity(unittest.TestCase):
    def test_channels_fidelity(self):
        p = 0.1  # Error probability
        fidelity_dephasing = average_gate_fidelity(DephasingChannel, p) # type: ignore
        fidelity_depolarizing = average_gate_fidelity(DepolarizingChannel, p) # type: ignore
        fidelity_amplitude_damping = average_gate_fidelity(AmplitudeDampingChannel, p) # type: ignore

        # Check if the fidelities are equal
        self.assertAlmostEqual(fidelity_dephasing, fidelity_depolarizing, places=5)
        self.assertAlmostEqual(fidelity_dephasing, fidelity_amplitude_damping, places=5)


class TestFunctions(unittest.TestCase):

    def setUp(self) -> None:
        # Create 3 quantum states
        rho0 = numpy.array([[1, 0], [0, 0]])
        rho1 = 1/2*numpy.array([[1, 1], [1, 1]])
        rho2 = numpy.array([[0, 0], [0, 1]])
        self.test_state = rho0
        self.states = [rho0, rho1, rho2]
        self.result = [1, .5, 0]

    def tearDown(self) -> None:
        return super().tearDown()

    def test_drift(self):
        for idx, elem in enumerate(self.states):
            with self.subTest(idx=idx):
                d = drift(self.test_state, elem)
                self.assertAlmostEqual(d, self.result[::-1][idx])

    def test_fidelity(self):
        for idx, elem in enumerate(self.states):
            with self.subTest(idx=idx):
                f = fidelity(self.test_state, elem)
                self.assertAlmostEqual(f, self.result[idx])


if __name__ == '__main__':
    unittest.main()
