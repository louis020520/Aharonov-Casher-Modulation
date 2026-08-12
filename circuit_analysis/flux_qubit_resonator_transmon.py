# Code for calculating the probe qubit-cavity coupling using a projected Hamiltonian
# (circuit used in the tomography paper)

import numpy as np
import scipy as sp
import scipy.sparse.linalg
import scipy.sparse as sparse
import matplotlib.pyplot as plt

print('The "RCCircuitCoup" module has been imported. This module allows to calculate the parallel and perpendicular coupling in a circuit of a flux qubit (probe) coupled to a resonator coupled to a transmon qubit. The method uses an Hamiltonian term projected into the probe qubit subspace.')

class RCCircuitCoup:
    def __init__(self, Ejp=121e9, Ejt=5e9, Cjp=8e-15, Ccp=5e-15, alphas=[0.4, 0.4, 1, 1], ng=0.25, flux=0.5,
                 Cr=100e-15, Lr=10e-9,  Cjt=4e-15, Cct=5e-15, Ct=40e-15, ncut=5):
        self.h = 6.626e-34
        self.hbar = 1.055e-34
        self.e_charge = 1.60218e-19

        self.Ejp = Ejp * self.h                 # Probe qubit Josephson energy
        self.Cjp = Cjp                          # Probe qubit Josephson capacitance
        self.Ccp = Ccp                          # Probe qubit coupling capacitance
        self.alphas = alphas                    # Asymmetry of probe qubit
        self.ng = ng                            # Reduced gate charge
        self.flux = flux                        # Flux through probe qubit
        self.Cr = Cr                            # Resonator capacitance
        self.Lr = Lr                            # Resonator inductance
        self.Ejt = Ejt * self.h                 # Target qubit Josephson energy
        self.Cjt = Cjt                          # Target qubit Josephson capacitance
        self.Cct = Cct                          # Target qubit coupling capacitance
        self.Ct = Ct                            # Target qubit capacitance
        self.ncut = ncut                        # Cut-off threshold for number basis

        self.init_operators()

    def _print_params(self):
        print(f'Ejp:    {self.Ejp}')
        print(f'Cjp:    {self.Cjp}')
        print(f'Ccp:    {self.Ccp}')
        print(f'alphas: {self.alphas}')
        print(f'ng:    {self.ng}')
        print(f'flux:  {self.flux}')
        print(f'Cr:    {self.Cr}')
        print(f'Lr:    {self.Lr}')
        print(f'Ejt:   {self.Ejt}')
        print(f'Cjt:   {self.Cjt}')
        print(f'Cct:   {self.Cct}')
        print(f'Ct:   {self.Ct}')

    def init_operators(self):
        self.I_cb = sp.sparse.diags(np.ones(2 * self.ncut + 1, dtype=np.clongdouble))   # Identity for qubit (charge basis)
        
        self.q_op_cb = sp.sparse.diags(-2 * self.e_charge * np.arange(-self.ncut, self.ncut + 1, dtype=np.clongdouble))           # Charge operator (charge basis)
        self.ng_op_cb = -2 * self.e_charge * self.ng * self.I_cb
        self.e_iphi_op_cb = sp.sparse.diags(np.ones(2 * self.ncut, dtype=np.clongdouble), offsets=1)                              # e^{i \phi} operator (charge basis)
        
        self.q1_p = self.tensor3(self.q_op_cb, self.I_cb, self.I_cb)
        self.q2_p = self.tensor3(self.I_cb, self.q_op_cb, self.I_cb)
        self.q3_p = self.tensor3(self.I_cb, self.I_cb, self.q_op_cb + self.ng_op_cb)

        # not really needed:
        self.q1_q1_p = self.tensor3(self.q_op_cb @ self.q_op_cb, self.I_cb, self.I_cb)
        self.q1_q2_p = self.tensor3(self.q_op_cb, self.q_op_cb, self.I_cb)
        self.q1_q3_p = self.tensor3(self.q_op_cb, self.I_cb, self.q_op_cb + self.ng_op_cb)
        self.q2_q2_p = self.tensor3(self.I_cb, self.q_op_cb @ self.q_op_cb, self.I_cb)
        self.q2_q3_p = self.tensor3(self.I_cb, self.q_op_cb, self.q_op_cb + self.ng_op_cb)
        self.q3_q3_p = self.tensor3(self.I_cb, self.I_cb, (self.q_op_cb + self.ng_op_cb) @ (self.q_op_cb + self.ng_op_cb))


    def tensor3(self, op1, op2, op3):
        return sparse.kron(sparse.kron(op1, op2), op3)

    def tensor4(self, op1, op2, op3, op4):
        return sparse.kron(sparse.kron(sparse.kron(op1, op2), op3), op4)

    def tensor5(self, op1, op2, op3, op4, op5):
        return sparse.kron(sparse.kron(sparse.kron(sparse.kron(op1, op2), op3), op4), op5)

    def hc(self, state):
        return np.conjugate(state).T

    # Set Capacitance matrix:
    
    def get_C_mat_in(self):
        
        self.C_mat = [
            [(self.alphas[2] + self.alphas[0]) * self.Cjp, 0, -self.alphas[0] * self.Cjp, 0, 0],
            [0, (self.alphas[3] + self.alphas[1]) * self.Cjp, -self.alphas[1] * self.Cjp, 0, 0],
            [-self.alphas[0] * self.Cjp, -self.alphas[1] * self.Cjp, (self.alphas[0] + self.alphas[1]) * self.Cjp + self.Ccp, -self.Ccp, 0],
            [0, 0, -self.Ccp, self.Cr + self.Ccp + self.Cct, -self.Cct],
            [0, 0, 0, -self.Cct, self.Cjt + self.Cct + self.Ct]
        ]

        self.C_mat_in = np.linalg.inv(self.C_mat)

        return self.C_mat_in

    # Derive Hamiltonian for the probe and diagonalise it:

    def kin_p(self):
        self.init_operators()
        self.get_C_mat_in()

        C_mat_in = self.C_mat_in

        kin_p = C_mat_in[0][0] * self.q1_q1_p
        kin_p += C_mat_in[1][1] * self.q2_q2_p
        kin_p += C_mat_in[2][2] * self.q3_q3_p
        kin_p += 2 * C_mat_in[0][1] * self.q1_q2_p
        kin_p += 2 * C_mat_in[0][2] * self.q1_q3_p
        kin_p += 2 * C_mat_in[1][2] * self.q2_q3_p

        kin_p *= 0.5

        return kin_p

    def pot_p(self):
        self.init_operators()

        pot_p = -self.Ejp * 0.5 * self.alphas[2] * self.tensor3(self.e_iphi_op_cb + self.e_iphi_op_cb.T, self.I_cb, self.I_cb)
        pot_p += -self.Ejp * 0.5 * self.alphas[0] * self.tensor3(self.e_iphi_op_cb.T, self.I_cb, self.e_iphi_op_cb)
        pot_p += -self.Ejp * 0.5 * self.alphas[0] * self.tensor3(self.e_iphi_op_cb, self.I_cb, self.e_iphi_op_cb.T)
        pot_p += -self.Ejp * 0.5 * self.alphas[1] * np.exp(2j * np.pi * self.flux) * self.tensor3(self.I_cb, self.e_iphi_op_cb, self.e_iphi_op_cb.T)
        pot_p += -self.Ejp * 0.5 * self.alphas[1] * np.exp(-2j * np.pi * self.flux) * self.tensor3(self.I_cb, self.e_iphi_op_cb.T, self.e_iphi_op_cb)
        pot_p += -self.Ejp * 0.5 * self.alphas[3] * self.tensor3(self.I_cb, self.e_iphi_op_cb + self.e_iphi_op_cb.T, self.I_cb)
        pot_p += self.Ejp * sum(self.alphas) * self.tensor3(self.I_cb, self.I_cb, self.I_cb)

        return pot_p

    def get_H_p(self):
        self.H_p = self.kin_p() + self.pot_p()
        # self.H_p.eliminate_zeros()

        return self.H_p
 
    def diagonalise_p(self, update=False):
        if update:
            self.get_H_p()
        else:
            try:
                self.H_p
            except AttributeError:
                self.get_H_p()

        evals_p, evecs_p = sparse.linalg.eigs(
            self.H_p, 10, which='SR'
        )
        evecs_p = evecs_p.T

        args = np.argsort(evals_p)
        self.evals_p = evals_p[args]
        self.evecs_p = evecs_p[args]

        return self.evals_p, self.evecs_p
 
    def init_probe_states(self, update=False):
        if update:
            self.diagonalise_p(update=True)
        else:
            try:
                self.evecs_p
            except AttributeError:
                self.diagonalise_p()

        self.probe_0 = self.evecs_p[0]
        self.probe_1 = self.evecs_p[1]
        self.probe_minus = 2**-0.5 * (self.probe_0 - self.probe_1)
        self.probe_plus = 2**-0.5 * (self.probe_0 + self.probe_1)

    def _plot_probe_states(self):
        self.init_probe_states(update=True)

        plt.figure(figsize=(10, 7))
        plt.title(f'Probe State: |0>, ng: {self.ng}', size=20)
        plt.plot(np.real(self.probe_0))
        plt.plot(np.imag(self.probe_0))
        plt.show()

        plt.figure(figsize=(10, 7))
        plt.title(f'Probe State: |1>, ng: {self.ng}', size=20)
        plt.plot(np.real(self.probe_1))
        plt.plot(np.imag(self.probe_1))
        plt.show()

        plt.figure(figsize=(10, 7))
        plt.title(f'Probe State: |+>, ng: {self.ng}', size=20)
        plt.plot(np.real(self.probe_plus))
        plt.plot(np.imag(self.probe_plus))
        plt.show()

        plt.figure(figsize=(10, 7))
        plt.title(f'Probe State: |->, ng: {self.ng}', size=20)
        plt.plot(np.real(self.probe_minus))
        plt.plot(np.imag(self.probe_minus))
        plt.show()

    def calc_probe_freq(self, update=False):
        if update:
            self.get_H_p()
            self.init_probe_states(update=True)
        else:
            try:
                self.H_p
            except AttributeError:
                self.H_p = self.get_H_p()

            try:
                self.probe_0
                self.probe_1
            except AttributeError:
                self.init_probe_states()

        return self.hc(self.probe_1) @ self.H_p @ self.probe_1 - self.hc(self.probe_0) @ self.H_p @ self.probe_0


    # Derive couplings:
    
    def get_H_p_coupling(self):
        self.init_operators()
        self.get_C_mat_in()

        invCm = self.C_mat_in

        Cr_renorm = 1/(invCm[3][3])
        Zr_renorm = np.sqrt(self.Lr / Cr_renorm)
        coeff = np.sqrt(self.hbar / (2 * Zr_renorm))

        self.H_p_coupling = 2 * invCm[0][3] * self.q1_p
        self.H_p_coupling += 2 * invCm[1][3] * self.q2_p
        self.H_p_coupling += 2 * invCm[2][3] *  self.q3_p

        self.H_p_coupling *= 0.5*coeff 

        return self.H_p_coupling

    def get_H_pt_coupling(self):
        self.init_operators()
        self.get_C_mat_in()

        invCm = self.C_mat_in

        coeff_pt = 2 * self.e_charge

        self.H_pt_coupling = 2 * self.invCm[0][4] * self.q1_p
        self.H_pt_coupling += 2 * self.invCm[1][4] * self.q2_p
        self.H_pt_coupling += 2 * self.invCm[2][4] * self.q3_p

        self.H_pt_coupling *= 0.5*coeff_pt

        return self.H_pt_coupling


    def calc_g_parr(self, update=False):
        if update:
            self.get_H_p()
            self.init_probe_states(update=True)
        else:
            try:
                self.H_p
            except AttributeError:
                self.H_p = self.get_H_p()

            try:
                self.probe_minus
                self.probe_plus
            except AttributeError:
                self.init_probe_states()
        
        self.get_H_p_coupling()

        g_parr = self.hc(self.probe_minus) @ self.H_p_coupling @ self.probe_plus

        return g_parr

    def calc_g_perp(self, update=False):
        if update:
            self.get_H_p()
            self.init_probe_states(update=True)
        else:
            try:
                self.H_p
            except AttributeError:
                self.H_p = self.get_H_p()

            try:
                self.probe_0
                self.probe_1
            except AttributeError:
                self.init_probe_states()

        self.get_H_p_coupling()

        g_perp = self.hc(self.probe_0) @ self.H_p_coupling @ self.probe_1

        return g_perp


    def calc_g_pt_parr(self, update=False):
        if update:
            self.get_H_p()
            self.init_probe_states(update=True)
        else:
            try:
                self.H_p
            except AttributeError:
                self.H_p = self.get_H_p()

            try:
                self.probe_minus
                self.probe_plus
            except AttributeError:
                self.init_probe_states()
        
        self.get_H_pt_coupling()

        g_parr = self.hc(self.probe_minus) @ self.H_pt_coupling @ self.probe_plus

        return g_parr

    def calc_g_pt_perp(self, update=False):
        if update:
            self.get_H_p()
            self.init_probe_states(update=True)
        else:
            try:
                self.H_p
            except AttributeError:
                self.H_p = self.get_H_p()

            try:
                self.probe_0
                self.probe_1
            except AttributeError:
                self.init_probe_states()

        self.get_H_pt_coupling()

        g_perp = self.hc(self.probe_0) @ self.H_pt_coupling @ self.probe_1

        return g_perp


