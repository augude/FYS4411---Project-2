# Using the VQE algorithm to estimate the ground state energy of many-body Hamiltonians

This repository contains the source code for Project 2 in FYS4411/Project 1 in FYS5419. 

## Description and code organization
The project aims to use the variational quantum eigensolver to find the ground state energy of many-body Hamiltonians. We study both simple toy-models in 2 and 4 dimensions, as well as the Lipkin model with total spin $J = 1$ and $J = 2$.

We have written our quantum computer simulation in Python, which can be found in ``src/qc.py``. The calculations of the ground state energies are in separate notebooks for each system in the folder ``Hamiltonians``.
We have also used IBM's quantum devices to calculate the energy. The code used for this can be found in ``src/run_qiskit.py`` and can be executed at https://quantum-computing.ibm.com/.

The data we have produced can be found in CSV-files in the folder ``data`` and the folder ``plots`` contains the Tikz code used to generate our plots. 

## References 
Course page for FYS4411: https://www.uio.no/studier/emner/matnat/fys/FYS4411 

Course page for FYS5419: https://www.uio.no/studier/emner/matnat/fys/FYS5419/index-eng.html

Project description: https://github.com/CompPhysics/QuantumComputingMachineLearning/blob/gh-pages/doc/Projects/2023/Project1/pdf/Project1.pdf

