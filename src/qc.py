import numpy as np 

class One_qubit:
    def __init__(self):
        self.state = np.zeros(2, dtype=np.complex_)
        self.I = np.eye(2)
        self.Z = np.array([[1, 0], [0, -1]])
        self.X = np.array([[0, 1], [1, 0]])
        self.Y = np.array([[0, -1j], [1j, 0]])
        self.H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    def set_state(self, state):
        if abs(np.linalg.norm(state) - 1) > 1e-10:
            raise ValueError("The state vector must be normalized.")
        self.state = state

    def apply_hadamard(self):
        self.state = np.dot(self.H, self.state)
        return self.state

    def apply_x(self):
        self.state = np.dot(self.X, self.state)
        return self.state

    def apply_y(self):
        self.state = np.dot(self.Y, self.state)
        return self.state

    def apply_z(self):
        self.state = np.dot(self.Z, self.state)
        return self.state

    def measure(self, num_shots=1):
        prob = np.abs(self.state)**2
        possible = np.arange(len(self.state)) #possible measurement outcomes
        outcome = np.random.choice(possible, p=prob, size = num_shots) #measurement outcome
        self.state = np.zeros_like(self.state) #set state to the measurement outcome
        self.state[outcome[-1]] = 1
        return outcome
    
    def rotate_x(self, theta):
        # implement rotation around x axis
        Rx = np.cos(theta/2) * self.I - 1j * np.sin(theta/2) * self.X
        self.state = np.dot(Rx, self.state)

    def rotate_y(self, phi):    
        # implement rotation around y axis
        Ry = np.cos(phi/2) * self.I - 1j * np.sin(phi/2) * self.Y
        self.state = np.dot(Ry, self.state)
    
class Two_qubit(One_qubit):
    def __init__(self):   
        # how to acces the variables stored in the parent class
        super().__init__()
        self.state = np.zeros(4, dtype=np.complex_)
        self.CNOT01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        self.CNOT10 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
        self.SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

        # np.random.seed(0)

    def apply_cnot01(self):
        self.state = np.dot(self.CNOT01, self.state)
        return self.state
    
    def apply_cnot10(self):
        self.state = np.dot(self.CNOT10, self.state)
        return self.state
    
    def apply_swap(self):
        self.state = np.dot(self.SWAP, self.state)
        return self.state
    
    def apply_hadamard(self, qubit):
        if qubit == 0:
            self.state = np.kron(self.H, self.I).dot(self.state)
        elif qubit == 1:
            self.state = np.kron(self.I, self.H).dot(self.state)
        return self.state

    def apply_x(self, qubit):
        if qubit == 0:
            self.state = np.kron(self.X, self.I).dot(self.state)
        elif qubit == 1:
            self.state = np.kron(self.I, self.X).dot(self.state)    
        return self.state

    def apply_y(self, qubit):
        if qubit == 0:
            self.state = np.kron(self.Y, self.I).dot(self.state)
        elif qubit == 1:
            self.state = np.kron(self.I, self.Y).dot(self.state)
        return self.state
        
    def apply_z(self, qubit):
        if qubit == 0:
            self.state = np.kron(self.Z, self.I).dot(self.state)
        elif qubit == 1:
            self.state = np.kron(self.I, self.Z).dot(self.state)
        return self.state
        
    def rotate_x(self, theta, qubit):
        Rx = np.cos(theta/2) * self.I - 1j * np.sin(theta/2) * self.X
        if qubit == 0:
            self.state = np.kron(Rx, self.I).dot(self.state)
        elif qubit == 1:
            self.state = np.kron(self.I, Rx).dot(self.state)
        return self.state

    def rotate_y(self, phi, qubit):    
        # implement rotation around y axis
        Ry = np.cos(phi/2) * self.I - 1j * np.sin(phi/2) * self.Y
        if qubit == 0:
            self.state = np.kron(Ry, self.I).dot(self.state)
        elif qubit == 1:
            self.state = np.kron(self.I, Ry).dot(self.state)
        return self.state
