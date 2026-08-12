import numpy as np
import scipy.sparse as sp
import math
import scipy.sparse.linalg
import matplotlib.pyplot as plt

# Analysis of only the Bell circuit

print('BellCircuit class has been imported!')

class BellCircuit:
    def __init__(self, Ejp=6.5e9, Cjp=3e-15, Lp=400e-9, Cp=3e-15, Cg=3e-15, 
                 Cr=100e-15, Lr=10e-9, Ccp=5e-15, Cct=5e-15, alpha=1,
                 Ejt=5e9, Cjt=4e-15, Ct=40e-15, flux=0.5, ng=0.0, ncut=5):

        self.h = 6.626e-34
        self.hbar = 1.055e-34
        self.e = 1.60218e-19
        self.phi0 = self.h / (2 * self.e)  # flux quantum

        self.Ejp = Ejp * self.h      # Probe Josephson energy
        self.Cjp = Cjp               # Probe Josephson capacitance
        self.Lp = Lp                 # Superinductance
        self.Cp = Cp                 # Probe shunting capacitance
        self.Cg = Cg                 # Grounded capacitance
        self.flux = flux             # Flux through the probe
        self.ng = ng                 # Reduced gate charge
        self.alpha = alpha           # Asymmetric factor

        self.Cr = Cr                 # Resonator capacitance
        self.Lr = Lr                 # Resonator inductance
        self.Ccp = Ccp               # Probe-resonator coupling capacitance

        self.Ejt = Ejt               # Target Josephson energy
        self.Cjt = Cjt               # Target Josephson capacitance
        self.Ct = Ct                 # Target shunting capacitance
        self.Cct = Cct               # Resonator-target coupling capacitance

        self.ncut = ncut             # Cut-off threshold
        self.dim = 2 * ncut + 1      # Dimension
        self.basis = np.arange(-ncut, ncut + 1)

        self.init_ladder_operators()
        self.init_charge_operators()
        self.init_bosonic_operators()

        self.get_fluxonium_hamiltonian()
        self.diagonalise_fluxonium()

    def print_parameters(self):
        print(f'Ejp:     {self.Ejp}')
        print(f'Cjp:     {self.Cjp}')
        print(f'Lp:      {self.Lp}')
        print(f'Cp:      {self.Cp}')
        print(f'Cg:      {self.Cg}')
        print(f'flux:    {self.flux}')
        print(f'ng:      {self.ng}')
        print(f'alpha:   {self.alpha}')

        print(f'Lr:      {self.Lr}')
        print(f'Cr:      {self.Cr}')
        print(f'Ccp:     {self.Ccp}')

        print(f'Ejt:     {self.Ejt}')
        print(f'Cjt:     {self.Cjt}')
        print(f'Ct:      {self.Ct}')
        print(f'Cct:     {self.Cct}')

    def init_ladder_operators(self):
        # annihilation and creation operator for a single oscillator
        a = sp.diags(np.sqrt(np.arange(1, self.dim, dtype=float)), offsets=1, shape=(self.dim, self.dim), format='csc')
        adag = a.transpose().conj()
        return a, adag
    
    def get_C_mat_in(self):

        m = 2 * self.Cg / (2 * self.Cg + self.Ccp)
        n = self.Ccp / (2 * self.Cg + self.Ccp)

        self.C_mat = [
            [(1+self.alpha) * self.Cjp + m * self.Ccp, -0.5 * (1-self.alpha) * self.Cjp, -m * self.Ccp, 0],
            [-0.5 * (1-self.alpha) * self.Cjp, 0.5 * (self.Cg + 0.5 * (1+self.alpha) * self.Cjp + 2 * self.Cp), 0, 0],
            [-n * self.Ccp, 0, self.Cr + self.Ccp + self.Cct - n * self.Ccp, - self.Cct],
            [0, 0, -self.Cct, self.Cct + self.Ct + self.Cjt]
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
        self.sin_phi_op = (self.e_iphi_op - self.e_minus_iphi_op) / (2j)

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
        self.q2_op = sp.kron(self.I, self.q2)

        self.n2_op = sp.kron(self.I, self.n2)
        self.n2_n2_op = sp.kron(self.I, self.n2 @ self.n2)

        self.phi2_op = sp.kron(self.I, self.phi2)
        self.phi2_phi2_op = sp.kron(self.I, self.phi2 @ self.phi2)
        
    
    def get_fluxonium_hamiltonian(self):
        self.get_C_mat_in()
        self.init_ladder_operators()
        self.init_charge_operators()
        self.init_bosonic_operators()

        self.n1_n2_op = sp.kron(self.n_op + self.ng_op, self.n2)

        # kinetic term
        self.Ec1 = 0.5 * (self.e**2) * self.C_mat_in[0][0]
        self.Ec2 = 0.5 * (self.e**2) * self.C_mat_in[1][1]

        T_fluxonium = 4 * self.Ec1 * self.n1_n1_op
        T_fluxonium += 4 * self.Ec2 * self.n2_n2_op
        T_fluxonium += 2 * 2 * (self.e**2) * self.C_mat_in[0][1] * self.n1_n2_op
        
        # josephson term
        V_fluxonium = (1 + self.alpha) * self.Ejp * sp.kron(self.I, self.I)

        for n in range(10):
            coeff_c = (-1)**n / math.factorial(2*n) / (2**(2*n))
            V_fluxonium += -(1+self.alpha) * self.Ejp * coeff_c * sp.kron(self.cos_phi_op, self.phi2**(2*n))
        
        for n in range(10):
            coeff_s = (-1)**n / math.factorial(2*n + 1) / (2**(2*n + 1))
            V_fluxonium += -(1-self.alpha) * self.Ejp * coeff_s * sp.kron(self.sin_phi_op, self.phi2**(2*n + 1))
        

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
    
    # couplings

    def get_fluxonium_resonator_coupling(self):
        self.get_C_mat_in()
        self.init_charge_operators()
        self.init_bosonic_operators()

        Cr_renorm = 1 / self.C_mat_in[2][2]
        Zr_renorm = np.sqrt(self.Lr / Cr_renorm)
        coeff = np.sqrt(self.hbar / (2 * Zr_renorm))
        
        self.H_pr_coupling = self.C_mat_in[0][2] * self.q1_op
        self.H_pr_coupling += self.C_mat_in[2][0] * self.q1_op
        self.H_pr_coupling += self.C_mat_in[1][2] * self.q2_op
        self.H_pr_coupling += self.C_mat_in[2][1] * self.q2_op
        self.H_pr_coupling *= 0.5 * coeff  # probe-resonator coupling

        return self.H_pr_coupling

    def get_fluxonium_target_coupling(self):
        self.get_C_mat_in()
        self.init_charge_operators()
        self.init_bosonic_operators()

        coeff_pt = 2 * self.e

        self.H_pt_coupling = self.C_mat_in[0][3] * self.q1_op
        self.H_pt_coupling += self.C_mat_in[3][0] * self.q1_op
        self.H_pt_coupling += self.C_mat_in[1][3] * self.q2_op
        self.H_pt_coupling += self.C_mat_in[3][1] * self.q2_op
        self.H_pt_coupling *= 0.5 * coeff_pt  # probe-target coupling

        return self.H_pt_coupling

    def calc_g_perp(self, update=False):
        if update:
            self.get_fluxonium_hamiltonian()
            self.get_fluxonium_initial_states(update=True)
        else:
            try:
                self.H_fluxonium
            except AttributeError:
                self.H_fluxonium = self.get_fluxonium_hamiltonian()
            
            try:
                self.probe_0
                self.probe_1
            except AttributeError:
                self.get_fluxonium_initial_states()

        self.get_fluxonium_resonator_coupling()
        g_perp = self.hc(self.probe_0) @ self.H_pr_coupling @ self.probe_1  # probe-resonator transverse

        return (g_perp) * 1e-6 / self.h

    def calc_g_parr(self, update=False):
        if update:
            self.get_fluxonium_hamiltonian()
            self.get_fluxonium_initial_states(update=True)
        else:
            try:
                self.H_fluxonium
            except AttributeError:
                self.H_fluxonium = self.get_fluxonium_hamiltonian()
            
            try:
                self.probe_minus
                self.probe_plus
            except AttributeError:
                self.get_fluxonium_initial_states()

        self.get_fluxonium_resonator_coupling()
        g_parr = self.hc(self.probe_minus) @ self.H_pr_coupling @ self.probe_plus  # probe-resonator parallel

        return (g_parr) * 1e-6 / self.h

    def calc_g_pt_perp(self, update=False):
        if update:
            self.get_fluxonium_hamiltonian()
            self.get_fluxonium_initial_states(update=True)
        else:
            try:
                self.H_fluxonium
            except AttributeError:
                self.H_fluxonium = self.get_fluxonium_hamiltonian()
            
            try:
                self.probe_0
                self.probe_1
            except AttributeError:
                self.get_fluxonium_initial_states()

        self.get_fluxonium_target_coupling()
        g_pt_perp = self.hc(self.probe_0) @ self.H_pt_coupling @ self.probe_1  # probe-target transverse

        return (g_pt_perp) * 1e-6 / self.h

    def calc_g_pt_parr(self, update=False):
        if update:
            self.get_fluxonium_hamiltonian()
            self.get_fluxonium_initial_states(update=True)
        else:
            try:
                self.H_fluxonium
            except AttributeError:
                self.H_fluxonium = self.get_fluxonium_hamiltonian()
            
            try:
                self.probe_minus
                self.probe_plus
            except AttributeError:
                self.get_fluxonium_initial_states()

        self.get_fluxonium_target_coupling()
        g_pt_parr = self.hc(self.probe_minus) @ self.H_pt_coupling @ self.probe_plus  # probe-target parallel
        
        return (g_pt_parr) * 1e-6 / self.h