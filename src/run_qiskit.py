"""
Based on code example from IBM Quantum. 
Note: this will not run on a local machine, but requires an IBM Quantum account. 
We have used the IBM Quantum Lab to run this code and copied the results to the data folder.

"""

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit, Aer
from qiskit.algorithms import VQE
from qiskit.opflow import X, Z, I, Y, AerPauliExpectation, PauliSumOp
from qiskit.circuit import Parameter
from qiskit.circuit.library import EfficientSU2
from qiskit.algorithms.optimizers import ADAM, L_BFGS_B, POWELL

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.providers.ibmq import least_busy
from qiskit.utils import QuantumInstance
import numpy as np
from utils import write_to_csv

def Hamiltonian(v):
    """
    Define the Hamiltonain whose ground state energy we want to find.
    """
    H = PauliSumOp.from_list([('Z', -1), ('X', -v)])
    return H

service = QiskitRuntimeService(
    channel='ibm_quantum'
)


backend = least_busy(service.backends(filters=lambda x: x.configuration().n_qubits >= 4 and 
                                                     not x.configuration().simulator and x.status().operational==True))
options = {
	'backend': backend.name, # string (required)
}


print(f"Running on {options['backend']}")
n_points = 10
opt = POWELL() #optimizer for VQE

vvals = np.arange(0, 2.0, 11)
energies = np.zeros(len(vvals))
for index, v in enumerate(vvals):
    print(f'Running for lambda: {v}')
    H = Hamiltonian(v)
    runtime_inputs = {
        # A parameterized quantum circuit preparing
        # the ansatz wavefunction for the
        # VQE. It is assumed that
        # all qubits are initially in
        # the 0 state.
        'ansatz': EfficientSU2(num_qubits=1, su2_gates=['rx', 'ry'], reps=1), # object (required)

        # Initial parameters of the ansatz.
        # Can be an array or
        # the string ``'random'`` to choose
        # random initial parameters.
        'initial_parameters': 'random', # [array,string] (required)

        # The Hamiltonian whose smallest eigenvalue
        # we're trying to find. Should
        # be PauliSumOp
        'operator': H, # object (required)

        # The classical optimizer used in
        # to update the parameters in
        # each iteration. Can be either
        # any of Qiskit's Optimizer classes.
        # If a dictionary, only SPSA
        # and QN-SPSA are supported and
        # the dictionary must specify the
        # name and options of the
        # optimizer, e.g. ``{'name': 'SPSA', 'maxiter':
        # 100}``.
        'optimizer': opt, # object (required)

        # A list or dict (with
        # strings as keys) of operators
        # of type PauliSumOp to be
        # evaluated at the final, optimized
        # state.
        'aux_operators': None, # array

        # Initial position of virtual qubits
        # on the physical qubits of
        # the quantum device. Default is
        # None.
        'initial_layout': None, # [null,array,object]

        # The maximum number of parameter
        # sets that can be evaluated
        # at once. Defaults to the
        # minimum of 2 times the
        # number of parameters, or 1000.
        # 'max_evals_grouped': None, # integer

        # Whether to apply measurement error
        # mitigation in form of a
        # complete measurement fitter to the
        # measurements. Defaults to False.
        # 'measurement_error_mitigation': False, # boolean

        # The number of shots used
        # for each circuit evaluation. Defaults
        # to 1024.
        'shots': 4096 # integer
    }

    job = service.run(
        program_id=f'vqe',
        options=options,
        inputs=runtime_inputs,
        instance='ibm-q/open/main'
    )

    # Job id
    print(job.job_id)
    # See job status
    print(job.status())


    # Get results
    result = job.result()
    energies[index] = result['eigenvalue']
    
write_to_csv(vvals, energies, header='v,energy', filename='..data/Lipkin_J=1_qiskit2.csv')