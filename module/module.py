import numpy
import cirq
import sympy
import scipy
import networkx
from scipy import sparse
from cirq.ops import raw_types
from typing import Iterable

__all__ = ["DepolarizingChannel", "DephasingChannel", "AmplitudeDampingChannel",
           "MyClass", "drift", "fidelity", "average_gate_fidelity"]

def average_gate_fidelity(channel: raw_types.Gate, p: float) -> float:
    """
    Calculate the average gate fidelity of a quantum channel using quantum process tomography.

    This function applies the specified quantum channel to each of the Pauli X, Y, and Z operators,
    simulates the resulting quantum states, and then calculates the average gate fidelity based on
    these simulations.

    Args:
        channel (raw_types.Gate): A quantum channel (a Cirq gate) that represents the noise model.
        p (float): The error probability parameter for the quantum channel.

    Returns:
        float: Average gate fidelity
    """

    # List of Pauli matrices
    paulis = [cirq.unitary(cirq.X), cirq.unitary(cirq.Y), cirq.unitary(cirq.Z)]
    
    # Initialize density matrix simulator
    simulator = cirq.DensityMatrixSimulator(split_untangled_states=True)
    
    # Initialize a single qubit
    qubit = cirq.LineQubit.range(1)
    
    F = 0
    # Iterate over each Pauli matrix
    for i in range(len(paulis)):
        # Compute eigenvalues and eigenvectors of the Pauli matrix
        eVals, eVecs = scipy.linalg.eigh(paulis[i]) # type: ignore
        
        # Iterate over the eigenvalues (and corresponding eigenvectors)
        for j in range(len(eVals)):
            # Prepare the initial state corresponding to the eigenvector
            initial_state = eVecs[:,j]

            # Create a circuit with the channel applied to the qubit
            circuit = cirq.Circuit(channel(p).on_each(*qubit)) # type: ignore

            # Simulate the circuit starting from the initial state
            result = simulator.simulate(circuit, initial_state=initial_state, qubit_order=qubit)

            # Extract the final state density matrix
            rho = result.final_density_matrix
        
            # Calculate the inner product (trace) of the final state with the Pauli matrix
            F += eVals[j] * numpy.trace(paulis[i] @ rho)

    # Calculate the average gate fidelity from the coefficients
    return (1/2 + 1/12 * F).real


def fidelity(A: numpy.ndarray, B: numpy.ndarray) -> float:
    """Computes the fidelity between two density matrices.

    Args:
        A (numpy.ndarray): Density matrix
        B (numpy.ndarray): Density matrix

    Returns:
        float: fidelity
    """
    Asqrtm = scipy.linalg.sqrtm(A) # type: ignore
    return (numpy.trace(scipy.linalg.sqrtm(Asqrtm@B@Asqrtm)).real)**2 # type: ignore


def dominant_eigenvector(A: numpy.ndarray) -> numpy.ndarray:
    """Returns the dominant eigenvector of a density matrix.

    Args:
        A (numpy.ndarray): Density matrix

    Returns:
        numpy.ndarray: Dominant eigenvector of A as column vector
    """
    # dimension of A
    dims = A.shape
    # diagonalize A
    eValues, eVectors = numpy.linalg.eigh(A)
    # find the position of the largest eigenvalue
    idx = numpy.argmax(eValues)
    # retrive the dominant eigenvector and reshape it to a column vector
    dom = numpy.reshape(eVectors[:, idx], (dims[0], 1))
    return dom


def drift(A: numpy.ndarray, B: numpy.ndarray) -> float:
    """Computes the coherent mismatch, also known as drift.

    Args:
        A (numpy.ndarray): Pure state as density matrix
        B (numpy.ndarray): Density matrix

    Returns:
        float: drift
    """
    dom = dominant_eigenvector(B)
    fidelity = abs(numpy.conj(dom.T)@A@dom)
    return 1 - float(fidelity)


class DepolarizingChannel(raw_types.Gate):
    def __init__(self, p: float) -> None:
        self._p = p

    def _num_qubits_(self) -> int:
        return 1

    def _mixture_(self):
        ps = [1.0 - self._p, self._p / 3, self._p / 3, self._p / 3]
        ops = [cirq.unitary(cirq.I), cirq.unitary(cirq.X),
               cirq.unitary(cirq.Y), cirq.unitary(cirq.Z)]
        return tuple(zip(ps, ops))

    def _has_mixture_(self) -> bool:
        return True

    def _circuit_diagram_info_(self, args) -> str:
        return f"Lambda_Dep({self._p})"


class DephasingChannel(raw_types.Gate):
    def __init__(self, p: float) -> None:
        self._p = p

    def _num_qubits_(self) -> int:
        return 1

    def _mixture_(self):
        ps = [1.0 - self._p, self._p]
        ops = [cirq.unitary(cirq.I), cirq.unitary(cirq.Z)]
        return tuple(zip(ps, ops))

    def _has_mixture_(self) -> bool:
        return True

    def _circuit_diagram_info_(self, args) -> str:
        return f"Lambda_Z({self._p})"

class AmplitudeDampingChannel(raw_types.Gate):
    def __init__(self, p: float) -> None:
        # Set the error probability such that the average gate fidelity 
        # is the same between all error-channels
        self._p = 4*(numpy.sqrt(1-p)+p-1)

    def _num_qubits_(self) -> int:
        return 1

    def _kraus_(self) -> Iterable[numpy.ndarray]:
        K1 = numpy.array([[1,0],
                          [0,numpy.sqrt(1-self._p)]])
        K2 = numpy.array([[0,numpy.sqrt(self._p)],
                          [0,0]])
        return [K1, K2]

    def _has_kraus_(self) -> bool:
        return True

    def _circuit_diagram_info_(self, args) -> str:
        return f"AD({self._p})"

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
        """Returns the MaxCut cost values of a graph

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
                     with_noise: raw_types.Gate = None) -> cirq.Circuit: # type: ignore
        """Creates the first iteration of the QAOA circuit

        Args:
            with_noise (raw_types.Gate, optional): Error channel to be
                appended after every gate. Defaults to None.

        Returns:
            cirq.Circuit: QAOA circuit
        """
        # Symbols for the rotation angles in the QAOA circuit.
        alpha = sympy.Symbol('alpha')
        beta = sympy.Symbol('beta')

        qubits = cirq.LineQubit.range(self.num_nodes)  # Create qubits

        circuit = cirq.Circuit()  # Initialize circuit
        for (u, v) in self.graph.edges:
            circuit.append(
                # This gate is equivalent to the RZZ-gate
                cirq.ops.ZZPowGate(
                    exponent=(alpha / scipy.pi), # type: ignore
                    global_shift=-.5)(qubits[u], qubits[v]) # type: ignore
                )
            if with_noise is not None:
                circuit.append(with_noise.on_each(qubits[u], qubits[v]))

        circuit.append(
            cirq.Moment(
                # This gate is equivalent to the RX-gate
                # That is why we multiply by two in the exponent
                cirq.ops.XPowGate(
                    exponent=(2 * beta / numpy.pi), # type: ignore
                    global_shift=-.5)(q) for q in qubits
            )
        )
        if with_noise is not None:
            # Make the error probability 10 times smaller than the error rate for two qubit gates.
            # Check the type of noise and append the appropriate channel.
            if isinstance(with_noise, DephasingChannel):
                # Append DepolarizingChannel after the RX-gate
                circuit.append(DepolarizingChannel(p=(with_noise._p / 10)).on_each(*qubits))
            elif isinstance(with_noise, AmplitudeDampingChannel):
                # Append AmplitudeDampingChannel instead of DepolarizingChannel after the RX-gate
                circuit.append(AmplitudeDampingChannel(p=with_noise._p / 10).on_each(*qubits))
        return circuit

    def simulate_qaoa(self,
                      params: tuple,
                      with_noise: raw_types.Gate = None) -> numpy.ndarray: # type: ignore
        """Simulates the p=1 QAOA circuit of a graph

        Args:
            params (tuple): Variational parameters
            with_noise (raw_types.Gate, optional): Error channel to be
                appended fter every gate. Defaults to None.

        Returns:
            numpy.ndarray: Density matrix output
        """
        alpha, beta = params
        resolver = cirq.ParamResolver({'alpha': alpha, 'beta': beta})

        circuit = self.qaoa_circuit(with_noise=with_noise)

        # prepare initial state |++...+>
        initial_state = 1 / numpy.sqrt(2**self.num_nodes) * \
            numpy.ones(2**self.num_nodes)

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
                with_noise=args[0] # type: ignore
            )
        else:
            rho = self.simulate_qaoa(
                params=x
            )
        return numpy.trace(self.cost * rho).real

    def optimize_qaoa_with_vd(self, x: tuple, *args: tuple) -> float:
        """Optimization function for QAOA with Virtual Distillation
            that is compatible with Scipy optimize.

        Args:
            x (tuple): Variational parameters
            *args (tuple): Error Channel
        Returns:
            float: Expectation value
        """
        expval = 0  # Initialize expval to a default value
        if args:
            error_channel = args[0]
            # Get the which type of noise we are dealing with
            noise_type = type(error_channel).__name__
            # Simulate QAOA with errors
            rho = self.simulate_qaoa(
                params=x,
                with_noise=error_channel # type: ignore
            )
            if noise_type == "DepolarizingChannel":
                # Error probability
                p = error_channel._p # type: ignore
                # Compute the mitigated expectation value
                expval = self.mitigated_cost(rho, p)
            elif noise_type == "DephasingChannel":
                # Compute the mitigated expectation value
                expval = self.mitigated_cost(rho, 0)
            elif noise_type == "AmplitudeDampingChannel":
                # Compute the mitigated expectation value
                expval = self.mitigated_cost_explicit(rho, error_channel) # type: ignore
        else:
            # Simulate QAOA without errors
            rho = self.simulate_qaoa(
                params=x
            )
            # Compute the mitigated expectation value
            expval = self.mitigated_cost(rho, 0)
        return expval.real

    def unmitigated_cost(self, rho: numpy.ndarray) -> float:
        """Calculates the unmitigated cost

        Args:
            rho (numpy.ndarray): Density matrix.

        Returns:
            float: Unmitigated cost
        """
        return numpy.trace(self.cost * rho).real
    
    def mitigated_cost_explicit(self, rho: numpy.ndarray, with_noise: raw_types.Gate = None) -> float: # type: ignore
        """Calculates the mitigated cost explicitly of a quantum state subject to a noise channel.
        The function applies virtual distillation to the given quantum state `rho`,
        simulates the effect of noise, and computes the mitigated cost.
        The mitigated cost is obtained using a symmetrized cost matrix and an observable
        for the ancilla qubit.

        Args:
            with_noise (raw_types.Gate): Noise channel (default is None).
            rho (numpy.ndarray): The input density matrix representing the quantum state before noise application.
            p (float, optional): The probability parameter for the amplitude damping channel (default is 0).

        Returns:
            mitigated cost (float): The mitigated cost after applying noise and virtual distillation to the quantum state.
        """
        # Run virtual distillation with channel noise applied
        rho_out = self.simulate_virtual_distillation(
            rho,
            with_noise=with_noise
        )
        
        # Retrieve the cost function matrix from the instance
        C = self.cost

        # Calculate the dimension of the system based on the number of nodes
        dim = 2**self.num_nodes

        # Construct a symmetrized cost matrix by using the Kronecker product
        # This represents the cost for both the system and an ancilla qubit
        C_2 = 1 / 2 * numpy.kron(numpy.ones(2), numpy.kron(C, numpy.ones(dim))) \
            + 1 / 2 * numpy.kron(numpy.ones(2), numpy.kron(numpy.ones(dim), C))

        # Define the Pauli X matrix
        x = numpy.array([[0, 1], [1, 0]])

        # Construct the observable for the ancilla qubit
        X_0 = sparse.kron(x, sparse.identity(dim**2))
        # Apply the observable to the output state from virtual distillation
        _X = (X_0 @ rho_out)

        # Calculate the expectation value of the numerator of the mitigated cost
        numerator = numpy.trace(C_2 * _X)
        # Calculate the expectation value of the denominator of the mitigated cost
        denominator = numpy.trace(_X)

        # Compute the mitigated cost by dividing the numerator by the denominator
        mitigated_cost = (numerator / denominator)
        return mitigated_cost.real

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
                expval_zz += (1 - 4 * p / 3)**2 * \
                    numpy.trace(zz.matrix(qubits)@rho_sq).real
            m_cost = 1 / 2 * (expval_zz - self.num_edges)
        return m_cost

    def virtual_distillation(self,
                             with_noise: raw_types.Gate = None) -> cirq.Circuit: # type: ignore
        """Creates the virtual distillation circuit for M=2

        Args:
            with_noise (raw_types.Gate, optional): Error channel to be
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

        return circuit

    def simulate_virtual_distillation(self,
                                      rho: numpy.ndarray,
                                      with_noise: raw_types.Gate = None) -> numpy.ndarray: # type: ignore
        """Simulates the virtual distillation process

        Args:
            rho (numpy.ndarray): Density matrix to be purified
            with_noise (raw_types.Gate, optional): Error channel to be
                appended after every gate. Defaults to None.

        Returns:
            cirq.DensityMatrixTrialResult: Result
        """
        shape = rho.shape
        dim = shape[0]

        if numpy.log2(dim) != self.num_nodes:
            assert("Error, dimensions are not correct")

        circuit = self.virtual_distillation(with_noise=with_noise)

        # Prepare the ancilla in the plus basis |+>
        ancilla = 1 / 2 * numpy.array([[1, 1], [1, 1]])

        # Prepare the total initial state
        initial_state = cirq.kron(ancilla, rho, rho)

        # Change data type to complex64
        initial_state = initial_state.astype('complex64')

        """
        cirq.validate_density_matrix requires all eigenvalues to be > -atol
        where atol = 1e-7. Because we can't give atol as an argument to the
        circuit simulator we have to approximate the density matrix to a valid
        one.
        """
        """
        atol = 1e-7
        eVals, eVecs = numpy.linalg.eigh(initial_state)
        if (eVals > -atol).all() == False:
            # Project the density matrix closer to the valid subspace
            eVals[eVals < -atol] = 0
            new_initial_state = eVecs@numpy.diag(eVals)@numpy.conj(eVecs.T)
            # Check that the new denstiy matrix is close to the original one
            # by computing the fidelity
            f = fidelity(new_initial_state, initial_state)
            if (1 - f) > atol:
                assert(
                    "Approximated density matrix is not close enough to the original one"
            else:
                # Replace the original density matrix with the new one
                initial_state = new_initial_state
        """
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
        X_0 = sparse.kron(x, sparse.identity(dim**2))

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
            rho (numpy.ndarray): input density matrix to virtual distillation.

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
