import numpy
import cirq
import sympy
import scipy
import networkx

__all__ = ["DepolarizingChannel", "DephasingChannel", "MyClass"]


class DepolarizingChannel(cirq.SingleQubitGate):
    def __init__(self, p: float) -> None:
        self._p = p

    def _mixture_(self):
        ps = [1.0 - 3 * self._p / 4, self._p / 4, self._p / 4, self._p / 4]
        ops = [cirq.unitary(cirq.I), cirq.unitary(cirq.X),
               cirq.unitary(cirq.Y), cirq.unitary(cirq.Z)]
        return tuple(zip(ps, ops))

    def _has_mixture_(self) -> bool:
        return True

    def _circuit_diagram_info_(self, args) -> str:
        return f"D({self._p})"


class DephasingChannel(cirq.SingleQubitGate):
    def __init__(self, p: float) -> None:
        self._p = p

    def _mixture_(self):
        ps = [1.0 - self._p / 2, self._p / 2]
        ops = [cirq.unitary(cirq.I), cirq.unitary(cirq.Z)]
        return tuple(zip(ps, ops))

    def _has_mixture_(self) -> bool:
        return True

    def _circuit_diagram_info_(self, args) -> str:
        return f"Lambda({self._p})"


class MyClass(object):
    def __init__(self, graph: networkx.Graph) -> None:
        """Init

        Args:
            graph (networkx.Graph): A Max-Cut graph
        """
        self.graph = graph
        self.num_nodes = len(graph.nodes)
        self.num_edges = len(graph.edges)
        self.cost = self.get_cost()

    def get_cost(self) -> numpy.ndarray:
        """
        Returns the MaxCut cost values of a graph

        Returns:
            numpy.ndarray: The cost values as an 1D-array
        """
        def product(*args, repeat=1):
            # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
            # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
            pools = [list(pool) for pool in args] * repeat
            result = [[]]
            for pool in pools:
                result = [x + [y] for x in result for y in pool]
            for prod in result:
                yield list(prod)

        # Number of edges
        M = self.num_edges
        # Number of nodes
        N = self.num_nodes
        # Adjacency matrix
        A = networkx.adjacency_matrix(self.graph).todense()

        # Generate a list of all possible n‐tuples of elements from {1,-1} and
        # organize them as a (2^n x n) matrix. In other words create all
        # possible solutions to the problem.
        s = numpy.array(list(product([1, -1], repeat=N)))

        # Construct the the cost function for Max Cut: C=1/2*Sum(Z_i*Z_j)-M/2
        # Note: This is the minimization version
        return 1 / 2 * (numpy.diag(s@numpy.triu(A)@s.T) - M)

    def qaoa_circuit(self,
                     meas: bool = True,
                     with_noise: cirq.SingleQubitGate = None) -> cirq.Circuit:
        """Creates the first iteration of the QAOA circuit

        Args:
            meas (bool, optional): Measurments at the end. Defaults to True.
            with_noise (cirq.SingleQubitGate, optional): Error channel to be
                appended after every gate. Defaults to None.

        Returns:
            cirq.Circuit: QAOA circuit
        """
        # Symbols for the rotation angles in the QAOA circuit.
        alpha = sympy.Symbol('alpha')
        beta = sympy.Symbol('beta')

        qubits = cirq.LineQubit.range(self.num_nodes)  # Create qubits

        circuit = cirq.Circuit()  # Initialize circuit
        circuit.append(cirq.H(q) for q in qubits)  # Add Hadamard

        if with_noise != None:
            circuit.append(with_noise.on_each(*qubits))

        for (u, v) in self.graph.edges:
            circuit.append(
                # This gate is equivalent to the RZZ-gate
                cirq.ops.ZZPowGate(
                    exponent=(alpha / numpy.pi),
                    global_shift=-.5
                )(qubits[u], qubits[v])
            )
            if with_noise != None:
                circuit.append(with_noise.on_each(qubits[u], qubits[v]))

        circuit.append(
            cirq.Moment(
                # This gate is equivalent to the RX-gate
                # That is why we multiply by two in the exponent
                cirq.ops.XPowGate(
                    exponent=(2 * beta / numpy.pi),
                    global_shift=-.5
                )(q) for q in qubits
            )
        )

        if with_noise != None:
            circuit.append(with_noise.on_each(*qubits))

        if meas:
            circuit.append(cirq.Moment(cirq.measure(q) for q in qubits))
        return circuit

    def simulate_qaoa(self,
                      params: tuple,
                      meas: bool = False,
                      with_noise: cirq.SingleQubitGate = None) -> numpy.ndarray:
        """Simulates the p=1 QAOA circuit of a graph

        Args:
            params (tuple): Variational parameters
            meas (bool): Measurments at the end. Defaults to True.
            with_noise (cirq.SingleQubitGate, optional): Error channel to be
                appended fter every gate. Defaults to None.

        Returns:
            numpy.ndarray: Density matrix output
        """
        alpha, beta = params
        resolver = cirq.ParamResolver({'alpha': alpha, 'beta': beta})

        circuit = self.qaoa_circuit(meas=meas, with_noise=with_noise)

        # prepare initial state |00...0>
        initial_state = numpy.zeros(2**self.num_nodes)
        initial_state[0] = 1

        # Density matrix simulator
        sim = cirq.DensityMatrixSimulator(
            split_untangled_states=True
        )

        # Simulate the QAOA
        result = sim.simulate(
            circuit,
            resolver,
            initial_state=initial_state,
            qubit_order=cirq.LineQubit.range(self.num_nodes)
        )
        return result.final_density_matrix

    def optimize_qaoa(self, x: tuple, *args: tuple) -> float:
        """Optimization function for QAOA that is compatible with
            Scipy optimize.

        Args:
            x (tuple): Variational parameters
            *args (tuple): Error Channel
        Returns:
            float: Expectation value
        """
        if args:
            rho = self.simulate_qaoa(
                params=x,
                with_noise=args[0]
            )
        else:
            rho = self.simulate_qaoa(
                params=x
            )
        return numpy.trace(self.cost * rho).real

    def unmitigated_cost(self, rho: numpy.ndarray) -> float:
        """Calculates the unmitigated cost

        Args:
            rho (numpy.ndarray): Density matrix.

        Returns:
            float: Unmitigated cost
        """
        return numpy.trace(self.cost * rho).real

    def mitigated_cost(self, rho: numpy.ndarray, p: float = 0) -> float:
        """Calculates the mitigated cost of virtual distillation

        Args:
            rho (numpy.ndarray): Input density matrix
            p (float): Error probability. Defaults to 0.

        Returns:
            float: Mitigated cost
        """
        # Change dtype to complex64
        if rho.dtype != 'complex64' or rho.dtype != 'complex128':
            rho = rho.astype('complex64')
        # Normalized square of density matrix
        rho_sq = rho@rho / numpy.trace(rho@rho)
        if p == 0:
            # This is mitigated cost for dephasing noise
            m_cost = numpy.trace(self.cost * rho_sq).real
        else:
            # This is the mitigated cost for depolarizing noise
            qubits = cirq.LineQubit.range(self.num_nodes)
            expval_zz = 0
            for u, v in self.graph.edges:
                zz = cirq.PauliString(cirq.Z(qubits[u])) \
                    * cirq.PauliString(cirq.Z(qubits[v]))
                expval_zz += (1 - p)**2 * \
                    numpy.trace(zz.matrix(qubits)@rho_sq).real
            m_cost = 1 / 2 * (expval_zz - self.num_edges)
        return m_cost

    def virtual_distillation(self,
                             meas: bool = True,
                             with_noise: cirq.SingleQubitGate = None) -> cirq.Circuit:
        """Creates the virtual distillation circuit for M=2

        Args:
            meas (bool, optional): Measurments at the end. Defaults to True.
            with_noise (cirq.SingleQubitGate, optional): Error channel to be
                appended after every gate. Defaults to None.

        Returns:
            cirq.Circuit: Virtual distillation circuit
        """

        # Create qubits
        qubits = cirq.LineQubit.range(2 * self.num_nodes + 1)

        # Initialize circuit
        circuit = cirq.Circuit()

        for q in range(self.num_nodes):
            circuit.append(cirq.FREDKIN(
                qubits[0], qubits[q + 1], qubits[self.num_nodes + (q + 1)]))
            if with_noise != None:
                circuit.append(with_noise.on_each(
                    qubits[0], qubits[q + 1], qubits[self.num_nodes + (q + 1)]))

        if meas:
            circuit.append(cirq.Moment(cirq.measure(q) for q in qubits))
        return circuit

    def simulate_virtual_distillation(self,
                                      rho: numpy.ndarray,
                                      meas: bool = False,
                                      with_noise: cirq.SingleQubitGate = None) -> numpy.ndarray:
        """Simulates the virtual distillation process

        Args:
            rho (numpy.ndarray): Density matrix to be purified
            meas (bool): Measurments at the end. Defaults to True.
            with_noise (cirq.SingleQubitGate, optional): Error channel to be
                appended after every gate. Defaults to None.

        Returns:
            cirq.DensityMatrixTrialResult: Result
        """
        shape = rho.shape
        dim = shape[0]

        if numpy.log2(dim) != self.num_nodes:
            assert("Error, dimensions are not correct")

        circuit = self.virtual_distillation(meas=meas, with_noise=with_noise)

        # Prepare the ancilla in the plus basis |+>
        ancilla = 1 / 2 * numpy.array([[1, 1], [1, 1]])

        # Prepare the total initial state
        initial_state = cirq.kron(ancilla, rho, rho)

        # Change data type to complex64
        initial_state = initial_state.astype('complex64')

        # Simulating with the density matrix simulator.
        sim = cirq.DensityMatrixSimulator()

        # Simulate Virtual distillation
        result = sim.simulate(
            circuit,
            initial_state=initial_state,
            qubit_order=cirq.LineQubit.range(2 * self.num_nodes + 1)
        )
        return result.final_density_matrix

    def unmitigated_variance(self, rho: numpy.ndarray) -> float:
        """Calculates the variance of the estimator for the unmitigated
        expectation value

        Args:
            rho (numpy.ndarray): Density matrix output from QAOA circuit

        Returns:
            float: Variance of the unmitigated estimator
        """
        C = self.cost
        return (numpy.trace(C**2 * rho) - numpy.trace(C * rho)**2).real

    def mitigated_variance(self, rho: numpy.ndarray) -> float:
        """Calculates the mitigated variance of the estimator

        Args:
            rho (numpy.ndarray): Density matrix output from the
                virtual distillation circuit

        Returns:
            float: variance of the mitigated estimator
        """
        # Costs
        C = self.cost

        # Dimension
        dim = 2**self.num_nodes

        # Symmeterized cost
        C_2 = 1 / 2 * numpy.kron(numpy.ones(2), numpy.kron(C, numpy.ones(dim))) \
            + 1 / 2 * numpy.kron(numpy.ones(2), numpy.kron(numpy.ones(dim), C))

        # Pauli X
        x = numpy.array([[0, 1], [1, 0]])

        # Observable for ancilla
        X_0 = scipy.sparse.kron(x, scipy.sparse.identity(dim**2))

        _X = (X_0@rho)
        # expectation value of numerator
        numerator = numpy.trace(C_2 * _X)
        # expectation value of denominator
        denominator = numpy.trace(_X)
        # variance of the numerator
        var_numerator = numpy.trace(C_2**2 * rho) - numerator**2
        # variance of the denominator
        var_denominator = 1 - denominator**2
        # covariance
        cov = numpy.trace(C_2 * rho) - numerator * denominator
        # variance of the estimator
        var_estimator = (var_numerator / denominator**2
                         + numerator**2 / denominator**4 * var_denominator
                         - 2 * numerator / denominator**3 * cov).real
        return var_estimator

    def ideal_mitigated_variance(self, rho: numpy.ndarray) -> float:
        """Calculates the theoretical idel mitigated variance of the estimator

        Args:
            rho (np.ndarray): input density matrix to virtual distillation.

        Returns:
            float: variance of the estimator

        """
        O = self.cost
        rho_sq = rho@rho  # rho squared
        var_est = numpy.trace(O**2 * rho) / (2 * numpy.trace(rho_sq)**2) \
            + numpy.trace(O * rho)**2 / (2 * numpy.trace(rho_sq)**2) \
            + numpy.trace(O * rho_sq)**2 / (numpy.trace(rho_sq)**4) \
            - 2 * numpy.trace(O * rho_sq) / (numpy.trace(rho_sq)**3) \
            * numpy.trace(O * rho)
        return var_est

    def ground_state(self) -> numpy.ndarray:
        """Creates a ground state for a given Max-Cut graph

        Returns:
            numpy.ndarray: Ground state
        """
        # Optimal cost
        cost_min = min(self.cost)
        # Find the args that corresponds to the optimal cost
        args = numpy.where(self.cost == cost_min)
        # Create the ideal state
        rho = numpy.zeros(2**self.num_nodes)
        for arg in args[0]:
            rho[arg] = 1
        return rho / numpy.sum(rho)

    def mixed_state(self, p: float = .5, beta: float = .1) -> numpy.ndarray:
        """Creates a mixed state between the ground state and a thermal state
        for a given graph

        Args:
            p (float): Probability of ground state. Defaults to .5.
            beta (float): Temperature. Defaults to 0.1.

        Returns:
            numpy.ndarray: Density matrix

        """
        return p * self.ground_state() + (1 - p) * self.thermal_state(beta=beta)

    def thermal_state(self, beta: float = .1) -> numpy.ndarray:
        """Creates a thermal state for a given graph

        Args:
            beta (float): Temperature. Defaults to 0.1

        Returns:
            numpy.ndarray: Thermal state
        """
        rho = numpy.exp(-beta * self.cost)
        return rho / numpy.sum(rho)
