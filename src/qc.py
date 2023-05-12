import numpy as np 

class Qubit:
    def __init__(self, N):
        self.N = N
        self.state = np.zeros(int(2**N), dtype=np.complex_)
        self.I = np.eye(2)
        self.Z = np.array([[1, 0], [0, -1]])
        self.X = np.array([[0, 1], [1, 0]])
        self.Y = np.array([[0, -1j], [1j, 0]])
        self.H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        self.S = np.array([[1, 0], [0, 1j]])
        self.CNOT01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        self.CNOT10 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
        self.SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


    def set_state(self, state):
        if abs(np.linalg.norm(state) - 1) > 1e-10:
            raise ValueError("The state vector must be normalized.")
        self.state = state

    def measure(self, num_shots=1):
        prob = np.abs(self.state)**2
        possible = np.arange(len(self.state)) #possible measurement outcomes
        outcome = np.random.choice(possible, p=prob, size = num_shots) #measurement outcome
        self.state = np.zeros_like(self.state) #set state to the measurement outcome
        self.state[outcome[-1]] = 1
        return outcome
    
    def Rx(self, theta):
        # implement rotation around x axis
        Rx = np.cos(theta/2) * self.I - 1j * np.sin(theta/2) * self.X
        return Rx

    def Ry(self, phi):    
        # implement rotation around y axis
        Ry = np.cos(phi/2) * self.I - 1j * np.sin(phi/2) * self.Y
        return Ry
    

def get_energy(angles, number_shots, unitaries,  prepare_state, constants, const_term=None, target = None):
    """
        Unitaries is a list of unitaries that are applied to the state
        constants is a list of constants that are multiplied with the expectation values
        const_term is a constant that is added to the expectation value (corresponds to the constant in front of the identity)
    """
    N = int(np.log2(unitaries[0].shape[0]))
    # print(N)
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
