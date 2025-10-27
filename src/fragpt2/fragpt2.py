#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:22:36 2023

@author: emielkoridon
"""

# This is purely a wrapper class around the functions found in the folder: perturbations

import numpy as np
import time

from pyscf import gto
from pyscf.fci.rdm import make_dm1234
from fragpt2.fragci import FragCI
from fragpt2.perturbations import fragpt2_disp, fragpt2_1ct, fragpt2_2ct, fragpt2_tt
from fragpt2.rdm.rdm_1ct import compute_five_rdm, get_1ct_rdms
from fragpt2.rdm.rdm_2ct import get_2ct_rdms
from fragpt2.rdm.rdm_tt import get_tt_rdms


def run_fragpt2(E_0, H_0, S_ovlp, H_prime_col):
    psi1_c, residuals, rank, s = np.linalg.lstsq((H_0 - E_0 * S_ovlp),
                                                 -H_prime_col, rcond=None)
    e_pt2 = np.einsum('n, n', H_prime_col,
                      psi1_c)
    return e_pt2, psi1_c


class FragPT2(FragCI):
    """Child of FragCI class. Supports perturbation calculations of
    dispersion, charge-transfer and TT terms."""

    def __init__(self, mol: gto.mole.Mole,
                 occ_idx, act_idx,
                 fragments,
                 mo_energies=None, mo_coeff=None, verbose=1):
        """
        Initialize Perturbation class.

        Args:
            mol (gto.mole.Mole): Full RHF object from pyscf

            occ_idx (list): Occupied indices that are always kept doubly occupied

            act_idx (list): Active indices ordered by fragment, then occupied - virtual

            fragments (tuple): Tuple (ncas,nelecas) for each fragment

            mo_energies (np.ndarray): MO energies

            mo_coeff (np.ndarray): MO coefficients

            verbose (Bool, optional): Control verbosity

        """
        super().__init__(mol, occ_idx, act_idx, fragments,
                         mo_energies=mo_energies, mo_coeff=mo_coeff, verbose=verbose)

        self.indices = (self.l1_idx, self.l2_idx)
        self.integrals = (self.c1, self.c2)

        self.E_0 = None
        self.rdms = None
        self.rdms5 = None
        self.ct_rdms = None
        self.ct2_rdms = None
        self.tt_rdms = None

    def run(self):
        "Run all perturbations"
        self.run_disp()
        self.run_1ct()
        self.run_2ct()
        self.run_tt()
        return (self.e_pt2_disp, self.e_pt2_1ct_AB + self.e_pt2_1ct_BA,
                self.e_pt2_2ct_AB + self.e_pt2_2ct_BA, self.e_pt2_tt)

    def run_disp(self):
        "Calculate Dispersion PT2"
        self.get_four_rdms()
        if self.E_0 is None:
            self.E_0 = self.h0_expval(w_core=False)
        H_0 = fragpt2_disp.get_h0(self.indices, self.rdms, self.integrals)
        S_ovlp = fragpt2_disp.get_ovlp(self.indices, self.rdms)
        H_prime_col = fragpt2_disp.get_h_prime(
            self.indices, self.rdms, self.integrals)
        self.e_pt2_disp, self.psi1_c_disp = run_fragpt2(
            self.E_0, H_0, S_ovlp, H_prime_col)
        return self.e_pt2_disp

    def run_1ct(self):
        "Calculate single charge-transfer PT2"
        self.get_1ct_rdms()
        if self.E_0 is None:
            self.E_0 = self.h0_expval(w_core=False)

        # -------- AB ----------
        H_0_AB = fragpt2_1ct.get_h0(
            'A', self.indices, self.rdms5, self.ct_rdms, self.integrals)
        S_ovlp_AB = fragpt2_1ct.get_ovlp('A', self.indices, self.ct_rdms)
        H_prime_col_AB = fragpt2_1ct.get_h_prime(
            'A', self.indices, self.ct_rdms, self.integrals)
        self.e_pt2_1ct_AB, self.psi1_c_1ct_AB = run_fragpt2(
            self.E_0, H_0_AB, S_ovlp_AB, H_prime_col_AB)

        # -------- BA ----------
        H_0_BA = fragpt2_1ct.get_h0(
            'B', self.indices, self.rdms5, self.ct_rdms, self.integrals)
        S_ovlp_BA = fragpt2_1ct.get_ovlp('B', self.indices, self.ct_rdms)
        H_prime_col_BA = fragpt2_1ct.get_h_prime(
            'B', self.indices, self.ct_rdms, self.integrals)
        self.e_pt2_1ct_BA, self.psi1_c_1ct_BA = run_fragpt2(
            self.E_0, H_0_BA, S_ovlp_BA, H_prime_col_BA)

        return self.e_pt2_1ct_AB, self.e_pt2_1ct_BA

    def run_2ct(self):
        "Calculate double charge-transfer PT2"
        self.get_four_rdms()
        self.get_2ct_rdms()

        # -------- AB ----------
        H_0_AB = fragpt2_2ct.get_h0(
            'A', self.indices, self.rdms, self.ct2_rdms, self.integrals)
        S_ovlp_AB = fragpt2_2ct.get_ovlp('A', self.indices, self.ct2_rdms)
        H_prime_col_AB = fragpt2_2ct.get_h_prime(
            'A', self.indices, self.ct2_rdms, self.integrals)
        self.e_pt2_2ct_AB, self.psi1_c_2ct_AB = run_fragpt2(
            self.E_0, H_0_AB, S_ovlp_AB, H_prime_col_AB)

        # -------- BA ----------
        H_0_BA = fragpt2_2ct.get_h0(
            'B', self.indices, self.rdms, self.ct2_rdms, self.integrals)
        S_ovlp_BA = fragpt2_2ct.get_ovlp('B', self.indices, self.ct2_rdms)
        H_prime_col_BA = fragpt2_2ct.get_h_prime(
            'B', self.indices, self.ct2_rdms, self.integrals)

        self.e_pt2_2ct_BA, self.psi1_c_2ct_BA = run_fragpt2(
            self.E_0, H_0_BA, S_ovlp_BA, H_prime_col_BA)

        return self.e_pt2_2ct_AB, self.e_pt2_2ct_BA

    def run_tt(self):
        "Calculate TT PT2"
        self.get_tt_rdms()
        H_0 = fragpt2_tt.get_h0(self.indices, self.rdms,
                                self.tt_rdms, self.integrals)
        S_ovlp = fragpt2_tt.get_ovlp(self.indices, self.tt_rdms)
        H_prime_col = fragpt2_tt.get_h_prime(
            self.indices, self.tt_rdms, self.integrals)
        self.e_pt2_tt, self.psi1_c_tt = run_fragpt2(
            self.E_0, H_0, S_ovlp, H_prime_col)
        return self.e_pt2_tt

    def get_four_rdms(self, again=False):
        if self.rdms is None or again:
            t1 = time.time()
            _, two_rdm_l1, three_rdm_l1, four_rdm_l1 = make_dm1234(
                'FCI4pdm_kern_sf', self.civec_l1, self.civec_l1, self.ncas_l1, self.nelec_l1)
            _, two_rdm_l2, three_rdm_l2, four_rdm_l2 = make_dm1234(
                'FCI4pdm_kern_sf', self.civec_l2, self.civec_l2, self.ncas_l2, self.nelec_l2)
            if self.verbose:
                print('computing 4-RDMs took:', time.time()-t1)
            self.rdms = (self.one_rdm_l1, two_rdm_l1, three_rdm_l1, four_rdm_l1,
                         self.one_rdm_l2, two_rdm_l2, three_rdm_l2, four_rdm_l2)

    def get_five_rdms(self, again=False):
        if self.rdms5 is None or again:
            self.get_four_rdms()
            t1 = time.time()
            five_rdm_l1 = compute_five_rdm(
                self.civec_l1, self.ncas_l1, self.nelec_l1)
            five_rdm_l2 = compute_five_rdm(
                self.civec_l2, self.ncas_l2, self.nelec_l2)
            if self.verbose:
                print('computing 5-RDMs took:', time.time()-t1)
            self.rdms5 = self.rdms[:4] + \
                (five_rdm_l1,) + self.rdms[4:] + (five_rdm_l2,)

    def get_1ct_rdms(self, again=False):
        if self.ct_rdms is None or again:
            self.get_five_rdms()
            t1 = time.time()
            self.ct_rdms = {}
            self.ct_rdms['A'] = get_1ct_rdms('A', self.rdms5)
            self.ct_rdms['B'] = get_1ct_rdms('B', self.rdms5)
            if self.verbose:
                print('computing CT RDMs took:', time.time()-t1)

    def get_2ct_rdms(self, again=False):
        if self.ct2_rdms is None or again:
            self.get_four_rdms()
            t1 = time.time()
            self.ct2_rdms = {}
            self.ct2_rdms['A'] = get_2ct_rdms('A', self.civec_l1, self.civec_l2, self.ncas_l1,
                                              self.nelec_l1, self.ncas_l2, self.ncas_l2)
            self.ct2_rdms['B'] = get_2ct_rdms('B', self.civec_l1, self.civec_l2, self.ncas_l1,
                                              self.nelec_l1, self.ncas_l2, self.ncas_l2)
            if self.verbose:
                print('computing 2CT RDMs took:', time.time()-t1)

    def get_tt_rdms(self, again=False):
        if self.tt_rdms is None or again:
            self.get_four_rdms()
            t1 = time.time()
            self.tt_rdms = {}
            self.tt_rdms['l1'] = get_tt_rdms(
                self.civec_l1, self.ncas_l1, self.nelec_l1)
            self.tt_rdms['l2'] = get_tt_rdms(
                self.civec_l2, self.ncas_l2, self.nelec_l2)
            if self.verbose:
                print('computing TT RDMs took:', time.time()-t1)


if __name__ == '__main__':
    from pyscf import gto, scf
    from fragpt2 import unpack_pyscf
    import matplotlib.pyplot as plt

    basis = 'unc-cc-pvdz'
    symmetry = 0
    cart = True

    mol = gto.Mole(atom='../../examples/water_ammonia/water-ammonia.xyz',
                   basis=basis, symmetry=symmetry, cart=cart)
    # mol = gto.Mole(atom='../../examples/biphenyl/dihedral_15.xyz',
    #                basis=basis, symmetry=symmetry, cart=cart)
    mol.max_memory = 8000

    mol.build()
    mf = scf.RHF(mol)
    mf.run()

    nao, mo_energies, mo_coeff = unpack_pyscf(
        "../../examples/water_ammonia/ribo.pyscf")
    # nao, mo_energies, mo_coeff = unpack_pyscf("../../examples/biphenyl/ribo_15.pyscf")

    # Water ammonia:
    # # for CAS(2,2)(2,2):
    # act_idx_l1_base1 = [4, 11]
    # act_idx_l2_base1 = [8, 27]

    # # # # for CAS(4,4)(4,4):
    act_idx_l1_base1 = [4, 5, 11, 12]
    act_idx_l2_base1 = [8, 9, 27, 28]

    # # # for CAS(2,2)(4,4):
    # act_idx_l1_base1 = [5, 11]
    # act_idx_l2_base1 = [8, 9, 27, 28]

    # # # for CAS(4,3)(4,4):
    # act_idx_l1_base1 = [4, 5, 11]
    # act_idx_l2_base1 = [8, 9, 27, 28]

    # # Biphenyl 15 deg:
    # # for CAS(8,8):
    # act_idx_l1_base1 = [26, 27, 42, 43]
    # act_idx_l2_base1 = [40, 41, 56, 57]

    act_idx_l1 = np.array(act_idx_l1_base1) - 1
    act_idx_l2 = np.array(act_idx_l2_base1) - 1
    act_idx = np.concatenate((act_idx_l1, act_idx_l2))

    # SET HERE THE SIZE OF THE ACTIVE SPACE
    ncas = len(act_idx)
    nelecas = ncas - 2 * (ncas // 2 % 2)
    nelec_l1 = nelecas // 2
    nelec_l2 = nelecas // 2
    ncas_l1 = ncas // 2
    ncas_l2 = ncas // 2

    # ncas = len(act_idx)
    # nelecas = 6
    # nelec_l1 = 2
    # nelec_l2 = 4
    # ncas_l1 = 2
    # ncas_l2 = 4

    print(f'l1: CAS({nelec_l1},{ncas_l1})')
    print(f'l2: CAS({nelec_l2},{ncas_l2})')

    occ_idx = [x for x in range(mol.nelectron//2) if x not in act_idx]

    nroots = 1

    casci = FragPT2(mol, occ_idx, act_idx,
                    ((ncas_l1, nelec_l1), (ncas_l2, nelec_l2)),
                    mo_energies, mo_coeff)

    casci.run_mean_field()
    casci.run_full_casci(nroots)
    casci.run_l1_casci(nroots)
    casci.run_l2_casci(nroots)
    casci.run_ct1_casci(nroots)
    casci.run_ct2_casci(nroots)
    e_tot, e_l1, e_l2 = casci.run_self_consistent_casci()

    e_corr_naive = casci.e_l1 + casci.e_l2 - 2 * mf.e_tot
    e_sumcorr = casci.e_l1 + casci.e_l2 - mf.e_tot

    print(f'Naive energy:            {casci.e_naive:.6f}')
    print(f'Sum of corr energies:    {e_sumcorr:.6f}')
    print(f'scf energy:              {e_tot:.6f}')
    print(f'exact energy:            {casci.e_full:.6f}')

    e_pt2_disp = casci.run_disp()
    # e_pt3, e_pt2 = casci.run_fragpt3()

    # print(f'embedding energy w pt3:  {e_tot + e_pt2 + e_pt3:.6f}')
    print(f'Pt2 disp correction:     {e_pt2_disp:.8f}')
    print(f'embedding energy wo pt3: {e_tot + e_pt2_disp:.6f}')
    # print(f'Pt3 correction:          {e_pt3:.8f}')

    # e_gev = casci.solve_gev()

    # print(f'GEV problem energy: {e_gev:.6f}')

    # psi_c = casci.psi1_c[1:].reshape((casci.ncas_l1**2, casci.ncas_l2**2))

    # plt.imshow(psi_c)
    # plt.colorbar()
    # plt.show()

    # plt.imshow(psi_c_norm)
    # plt.colorbar()
    # plt.show()

    # casci.run_ct1_casci(nroots)
    # casci.run_ct2_casci(nroots)

    e_ct1_ci = casci.e_ct1 - mf.e_tot
    e_ct2_ci = casci.e_ct2 - mf.e_tot

    print(f'CT1 CI energy:        {e_ct1_ci:.6f}')
    print(f'CT2 CI energy:        {e_ct2_ci:.6f}')

    e_pt2_ct1_1, e_pt2_ct1_2 = casci.run_1ct()

    print(f'CT1_1 PT2 energy:     {e_pt2_ct1_1:.6f}')
    print(f'CT1_2 PT2 energy:     {e_pt2_ct1_2:.6f}')

    e_pt2_tt = casci.run_tt()

    print(f'PT2 TT correction:    {e_pt2_tt:.12f}')

    e_pt2_ct2_1, e_pt2_ct2_2 = casci.run_2ct()

    print(f'CT2_1 PT2 energy:     {e_pt2_ct2_1:.12f}')
    print(f'CT2_2 PT2 energy:     {e_pt2_ct2_2:.12f}')

    e_pt2_ct1 = e_pt2_ct1_1 + e_pt2_ct1_2
    e_pt2_ct2 = e_pt2_ct2_1 + e_pt2_ct2_2

    print(
        f'Full embedding E:     {e_tot + e_pt2_disp + e_pt2_tt + (e_pt2_ct1 + e_pt2_ct2):.6f}')
