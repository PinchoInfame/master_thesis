import numpy as np
from stlpy.systems import LinearSystem
class ProductDynamicalSystem():
    def __init__(self):
        self.A = None
        self.B = None
        self.C = None
        self.D = None
        self.combined_system = None
    def __call__(self, subsystems):
        #extract matrices A, B, C, D for all subsystems
        subsystems_number = len(subsystems)
        if subsystems_number == 1:
            print("Define more than one dynamical subsystem")
            exit()
        A_subsystem = []
        B_subsystem = []
        C_subsystem = []
        D_subsystem = []
        for i in range(subsystems_number):
            A_subsystem.append(subsystems[i].A)
            B_subsystem.append(subsystems[i].B)
            C_subsystem.append(subsystems[i].C) 
            D_subsystem.append(subsystems[i].D)
        # Create block diagonal matrices for the combined system
        A_combined = A_subsystem[0]
        B_combined = B_subsystem[0]
        C_combined = C_subsystem[0]
        D_combined = D_subsystem[0]
        for i in range(len(A_subsystem)-1):
            A_combined = np.block([
            [A_combined, np.zeros((np.size(A_combined, axis=0), np.size(A_subsystem[i+1], axis=1)))],
            [np.zeros((np.size(A_subsystem[i+1], axis=0), np.size(A_combined, axis=1))), A_subsystem[i+1]]
            ])
            B_combined = np.block([
            [B_combined, np.zeros((np.size(B_combined, axis=0), np.size(B_subsystem[i+1], axis=1)))],
            [np.zeros((np.size(B_subsystem[i+1], axis=0), np.size(B_combined, axis=1))), B_subsystem[i+1]]
            ])
            C_combined = np.block([
            [C_combined, np.zeros((np.size(C_combined, axis=0), np.size(C_subsystem[i+1], axis=1)))],
            [np.zeros((np.size(C_subsystem[i+1], axis=0), np.size(C_combined, axis=1))), C_subsystem[i+1]]
            ])
            D_combined = np.block([
            [D_combined, np.zeros((np.size(D_combined, axis=0), np.size(D_subsystem[i+1], axis=1)))],
            [np.zeros((np.size(D_subsystem[i+1], axis=0), np.size(D_combined, axis=1))), D_subsystem[i+1]]
            ])
        # Define the concatenated system as a LinearSystem
        self.combined_system = LinearSystem(A_combined, B_combined, C_combined, D_combined)
        self.A = self.combined_system.A
        self.B = self.combined_system.B
        self.C = self.combined_system.C
        self.D = self.combined_system.D
        return self.combined_system
        