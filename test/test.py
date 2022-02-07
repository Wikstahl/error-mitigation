import unittest
import random
import networkx
import numpy
import cirq
from module import MyClass, DepolarizingChannel, DephasingChannel


class TestGraph(unittest.TestCase):
    def setUp(self) -> None:
        # Create a random graph
        graph = networkx.generators.random_graphs.erdos_renyi_graph(n=4, p=0.5)
        # Create object
        self.obj = MyClass(graph)
        self.num_nodes = len(graph.nodes)
        # Prepare a mixed state
        self.input = numpy.diag(self.obj.mixed_state())
        # Try 3 different noise levels
        self.p = [0, .25, .5]
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

    def test_mitigated_cost_deppolarizing(self):
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
                self.assertTrue(numpy.isclose(
                    mitigated_cost, predicted_mitigated_cost))

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
                self.assertTrue(numpy.isclose(
                    mitigated_cost, predicted_mitigated_cost))

    def test_mitigated_variance_dephasing(self):
        for p in self.p:
            with self.subTest(p=p):
                var_predicted = 1 / (1 - p)**(2 * self.num_nodes) \
                    * self.obj.ideal_mitigated_variance(self.input)
                # Run virtual distillation
                rho_out = self.obj.simulate_virtual_distillation(
                    self.input,
                    with_noise=DephasingChannel(p=p)
                )
                var_estimated = self.obj.mitigated_variance(rho_out)
                self.assertTrue(numpy.isclose(
                    var_predicted, var_estimated, rtol=1e-4, atol=1e-6))

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
        self.assertTrue(numpy.isclose(var_mitigated, var_unmitigated / 2))


if __name__ == '__main__':
    unittest.main()
