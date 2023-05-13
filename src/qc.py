import numpy as np 

class Qubit:
    """
    Class for a qubit simulator.
    """
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
    