import numpy as np
from src.qc import Qubit

def get_energy(angles, number_shots, unitaries,  prepare_state, constants, const_term=None, target = None):
    """
        Approximated the expectation value of a Hamiltonian with a quantum circuit.
        Unitaries is a list of unitaries that are applied to the state
        constants is a list of constants that are multiplied with the expectation values
        const_term is a constant that is added to the expectation value (corresponds to the constant in front of the identity)
    """
    N = int(np.log2(unitaries[0].shape[0]))
    qubit = Qubit(N)
    init_state = prepare_state(angles, N, target)
    measures = np.zeros((len(unitaries), number_shots))
    for index, U in enumerate(unitaries):
        qubit.set_state(init_state)
        qubit.state = U@qubit.state
        measure = qubit.measure(number_shots)
        measures[index] = measure
    exp_vals = np.zeros(len(measures)) 
    for index in range(len(exp_vals)):
        counts = [len(np.where(measures[index] == i)[0]) for i in range(2**N)] 
        for outcome, count in enumerate(counts):
            if outcome < 2**N//2:
                exp_vals[index] += count #the first half of the outcomes correspond to 0 in the first qubit
            elif outcome >= 2**N//2:
                exp_vals[index] -= count #the latter half of the outcomes correspond to 1 in the first qubit
    if const_term is None:
        exp_val = np.sum(constants * exp_vals) / number_shots
    else:
        exp_val = np.sum(constants * exp_vals) / number_shots + const_term
    return exp_val

def prepare_state_1D(angles, N, target = None):
    """
    Quantum circuit for preparing a hardware-efficient ansatz in 1D
    """
    qubit = Qubit(N)
    theta, phi = angles
    I, X, Y = qubit.I, qubit.X, qubit.Y
    state = np.array([1, 0])
    Rx = qubit.Rx(theta)
    Ry = qubit.Ry(phi)
    state = Ry @ Rx @ state
    if target is not None:
        state = target
    return state

def prepare_state_2D(angles, N, target = None):
    """
    Quantum circuit for preparing a hardware-efficient ansatz in 2D
    """
    theta0, phi0, theta1, phi1 = angles
    qubit = Qubit(N)
    CNOT01 = qubit.CNOT01
    state = [1, 0, 0, 0]
    rotate = np.kron(qubit.Rx(theta0)@qubit.Ry(phi0), qubit.Rx(theta1)@qubit.Ry(phi1))
    state = CNOT01 @ rotate @ state #entangle & rotate
    if target is not None:
        state = target
    return state

def prepare_state_coupled(angles, N, target = None):
    """
    Quantum circuit for preparing a the ansatz used in the coupeld spin mapping for the Lipkin model with J = 2
    """
    qubit = Qubit(N)
    I = qubit.I
    state = np.zeros(2**N)
    state[0] = 1
    def control_rot(theta):
        c = np.cos(theta/2)
        s = np.sin(theta/2)
        c_rot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, c, -s], [0, 0, s, c]])
        return c_rot
    rot = np.kron(qubit.Ry(angles[0]), I)
    state = rot@state
    cr1 = control_rot(angles[1])
    state = cr1@state
    
    if target is not None:
        state = target
    qubit.set_state(state)
    return qubit.state

def prepare_state_individual(angles, N, target = None):
    """
    Quantum circuit for preparing a hardware-efficient ansatz in 4D used in the individual spin mapping
    """
    qubit = Qubit(N)
    I, X, Y, CNOT01 = qubit.I, qubit.X, qubit.Y, qubit.CNOT01
    state = np.zeros(2**N)
    state[0] = 1

    angles_batches = [angles[i:i+2*N] for i in range(0, len(angles), 2*N)]
    rotations = []
    for angles in angles_batches:
        for i in range(0, len(angles)-1, 2):
            theta, phi = angles[i], angles[i+1]
            Rx = np.cos(theta/2) * I - 1j * np.sin(theta/2) * X
            Ry = np.cos(phi/2) * I - 1j * np.sin(phi/2) * Y
            rotations.append(Ry@Rx)

    for i in range(0, len(rotations), N):    
        rotate = np.kron(rotations[i], np.kron(rotations[i+1], np.kron(rotations[i+2], rotations[i+3])))
        state = rotate @ state
        state = np.kron(I, np.kron(I, CNOT01))@state 
        state = np.kron(I, np.kron(CNOT01, I))@state 
        state = np.kron(CNOT01, np.kron(I, I))@state 

    if target is not None:
        state = target
    qubit.set_state(state)
    return qubit.state