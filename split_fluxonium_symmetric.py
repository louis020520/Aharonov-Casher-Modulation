import numpy as np
import scipy.sparse as sp
import math
import scipy.sparse.linalg
import matplotlib.pyplot as plt

# Analysis of only the Bell circuit

print('BellCircuit class has been imported!')

class BellCircuit:
    def __init__(self, Ej=6.5e9, Cj=3e-15, Lp=400e-9, Cp=3e-15, Cg=3e-15, flux=0.5, ng=0.0, ncut=5):

        self.h = 6.626e-34
        self.hbar = 1.055e-34
        self.e = 1.60218e-19
        self.phi0 = self.h / (2 * self.e)  # flux quantum

        self.Ej = Ej * self.h      # Josephson energy
        self.Cj = Cj               # Josephson capacitance
        self.Lp = Lp               # Superinductance
        self.Cp = Cp               # Shunting capacitance
        self.Cg = Cg               # Grounded capacitance
        self.flux = flux           # Flux through the qubit
        self.ng = ng               # Reduced gate charge

        self.ncut = ncut           # Cut-off threshold
        self.dim = 2 * ncut + 1    # Dimension
        self.basis = np.arange(-ncut, ncut + 1)

        self.init_ladder_operators()
        self.init_charge_operators()
        self.init_bosonic_operators()

        self.get_fluxonium_hamiltonian()
        self.diagonalise_fluxonium()

    def print_parameters(self):
        print(f'Ej:     {self.Ej}')
        print(f'Cj:     {self.Cj}')
        print(f'Lp:     {self.Lp}')
        print(f'Cp:     {self.Cp}')
        print(f'Cg:     {self.Cg}')
        print(f'flux:   {self.flux}')
        print(f'ng:     {self.ng}')

    
    def init_ladder_operators(self):
        # annihilation and creation operator for a single oscillator
        a = sp.diags(np.sqrt(np.arange(1, self.dim, dtype=float)), offsets=1, shape=(self.dim, self.dim), format='csc')
        adag = a.transpose().conj()
        return a, adag
    
    def get_C_mat_in(self):

        self.C_mat = [
            [2 * self.Cj, 0],
            [0, 0.5 * (self.Cg + self.Cj + 2 * self.Cp)]
        ]
        self.C_mat_in = np.linalg.inv(self.C_mat)

        return self.C_mat_in
    
    def init_charge_operators(self):
        self.I = sp.diags(np.ones(self.dim), format='csc')  # identity operator

        self.n_op = sp.diags(self.basis, 0, format='csc')  # charge operator
        self.q_op = -2 * self.e * self.n_op
        self.ng_op = self.ng * self.I   # offset charge operator
        self.qg_op = -2 * self.e * self.ng_op

        self.e_iphi_op = sp.diags(np.ones(self.dim - 1), offsets=1, format='csc')  # raising operator for charge
        self.e_minus_iphi_op = sp.diags(np.ones(self.dim - 1), offsets=-1, format='csc')  # lowering operator for charge
        self.cos_phi_op = (self.e_iphi_op + self.e_minus_iphi_op) / 2

        # charge operator on first node, including offset charge
        self.q1_op = sp.kron(self.q_op + self.qg_op, self.I)
        self.n1_op = sp.kron(self.n_op + self.ng_op, self.I)
        
        # quadratic
        self.q1_q1_op = sp.kron((self.q_op + self.qg_op) @ (self.q_op + self.qg_op), self.I)
        self.n1_n1_op = sp.kron((self.n_op + self.ng_op) @ (self.n_op + self.ng_op), self.I)
        
    
    def hc(self, state):
        return np.conjugate(state).T
    
    
    def init_bosonic_operators(self):
        self.get_C_mat_in()

        # ladder operators
        self.b, self.bdag = self.init_ladder_operators()

        # external flux operator
        self.flux_op = 2 * np.pi * self.flux * self.I   

        # zero-point fluctuations
        self.EL = (self.phi0**2) / (4 * np.pi**2 * self.Lp)
        self.ECb = (self.e**2 / 2.0) * self.C_mat_in[1][1]

        self.n_zpf = (self.EL / (32.0 * self.ECb))**0.25
        self.phi_zpf = (2.0 * self.ECb / self.EL)**0.25

        # charge and phase operator on second node
        self.n2 = 1j * self.n_zpf * (self.b - self.bdag)
        self.phi2 = self.phi_zpf * (self.b + self.bdag)

        self.q2 = -2 * self.e * self.n2

        self.n2_op = sp.kron(self.I, self.n2)
        self.n2_n2_op = sp.kron(self.I, self.n2 @ self.n2)

        self.phi2_op = sp.kron(self.I, self.phi2)
        self.phi2_phi2_op = sp.kron(self.I, self.phi2 @ self.phi2)
        

    
    def get_fluxonium_hamiltonian(self):
        self.get_C_mat_in()
        self.init_ladder_operators()
        self.init_charge_operators()
        self.init_bosonic_operators()

        # kinetic term
        self.Ec1 = 0.5 * (self.e**2) * self.C_mat_in[0][0]
        self.Ec2 = 0.5 * (self.e**2) * self.C_mat_in[1][1]

        T_fluxonium = 4 * self.Ec1 * self.n1_n1_op
        T_fluxonium += 4 * self.Ec2 * self.n2_n2_op
        
        # josephson term
        V_fluxonium = 2 * self.Ej * sp.kron(self.I, self.I)

        for n in range(10):
            coeff = (-1)**n / math.factorial(2*n) / (2**(2*n))
            V_fluxonium += -2 * self.Ej * coeff * sp.kron(self.cos_phi_op, self.phi2**(2*n))
        
        #V_fluxonium = -self.Ej * 2 * sp.kron(self.cos_phi_op, self.I)
        #V_fluxonium += self.Ej * 0.25 * sp.kron(self.cos_phi_op, self.phi2 @ self.phi2)
        #V_fluxonium += -self.Ej * (1 / 192) * sp.kron(self.cos_phi_op, self.phi2 @ self.phi2 @ self.phi2 @ self.phi2)
        #V_fluxonium += self.Ej * 2 * sp.kron(self.I, self.I)

        # inductive term
        self.El = (self.phi0**2) / (4 * np.pi**2 * self.Lp)   # superinductor inductive energy
        V_fluxonium += 0.5 * self.El * sp.kron(self.I, (self.phi2 - self.flux_op) @ (self.phi2 - self.flux_op))

        self.H_fluxonium = T_fluxonium + V_fluxonium

        return self.H_fluxonium


    def diagonalise_fluxonium(self, update=False):
        if update:
            self.init_charge_operators()
            self.init_bosonic_operators()
            self.get_fluxonium_hamiltonian()
        else:
            try:
                self.H_fluxonium
            except AttributeError:
                self.get_fluxonium_hamiltonian()
        
        evals, evecs = sp.linalg.eigsh(self.H_fluxonium, 10, which='SA')
        args = np.argsort(evals)

        self.evals_fluxonium = evals[args]
        self.evecs_fluxonium = evecs[:, args]
        return self.evals_fluxonium, self.evecs_fluxonium
    
    def get_fluxonium_initial_states(self, update=False):
        if update:
            self.diagonalise_fluxonium(update=True)
        else:
            try:
                self.evecs_fluxonium
            except:
                self.diagonalise_fluxonium()
        
        self.probe_0 = self.evecs_fluxonium[:, 0]   # probe zero state
        self.probe_1 = self.evecs_fluxonium[:, 1]   # probe one state

        self.probe_minus = 2**-0.5 * (self.probe_0 - self.probe_1)  # minus
        self.probe_plus = 2**-0.5 * (self.probe_0 + self.probe_1)  # plus

    def get_fluxonium_frequency(self, update=False):
        if update:
            self.diagonalise_fluxonium(update=True)
        else:
            try:
                self.evals_fluxonium
            except AttributeError:
                self.diagonalise_fluxonium()
        
        self.E1_fluxonium = self.evals_fluxonium[1]
        self.E0_fluxonium = self.evals_fluxonium[0]

        return (self.E1_fluxonium - self.E0_fluxonium) * 1e-9 / self.h