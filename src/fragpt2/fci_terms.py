#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:06:44 2023

@author: emielkoridon
"""


import numpy as np

from pyscf import gto, scf, fci
from pyscf.fci.direct_spin0 import make_rdm12
from pyscf.fci.direct_spin1 import make_rdm12s
from fragpt2.fragpt2 import FragPT2


def build_local_tensor(tensor, indices):
    local_tensor = tensor[np.ix_(*indices)]
    return local_tensor


def h_eff(one_body_integrals, two_body_integrals):
    return one_body_integrals - np.einsum('prrq->pq', two_body_integrals)


def fci_full(c1, c2, ncas, nelecas):
    cisolver = fci.direct_spin1.FCI()
    cisolver.max_cycle = 100
    cisolver.conv_tol = 1e-8
    return cisolver.kernel(c1, c2, ncas, nelecas)


class FCI_terms(FragPT2):
    def __init__(self, mf: scf.hf.RHF,
                 occ_idx, act_idx,
                 fragments,
                 mo_energies=None, mo_coeff=None, verbose=0):

        super().__init__(mf, occ_idx, act_idx, fragments,
                         mo_energies=mo_energies, mo_coeff=mo_coeff, verbose=verbose)

        nroots = 1
        # self.run_full_casci(nroots)
        self.run_l1_casci(nroots)
        self.run_l2_casci(nroots)
        self.run_ct1_casci(nroots)
        self.run_ct2_casci(nroots)

        self.e_tot, _, _ = self.run_self_consistent_casci()

        self.eci_full, self.fcivec = fci_full(
            self.c1, self.c2, self.ncas, self.nelecas)

        self.one_rdm_full, self.two_rdm_full = make_rdm12(
            self.fcivec, self.ncas, self.nelecas)
        _, self.two_rdm_reord_full = make_rdm12(
            self.fcivec, self.ncas, self.nelecas, reorder=False)

        self.c1_pure_l1 = build_local_tensor(self.c1, [self.l1_idx]*2)
        self.c2_pure_l1 = build_local_tensor(self.c2, [self.l1_idx]*4)
        self.one_rdm_l1_pure = build_local_tensor(
            self.one_rdm_full, [self.l1_idx]*2)
        self.two_rdm_l1_pure = build_local_tensor(
            self.two_rdm_full, [self.l1_idx]*4)
        self.c1_pure_l2 = build_local_tensor(self.c1, [self.l2_idx]*2)
        self.c2_pure_l2 = build_local_tensor(self.c2, [self.l2_idx]*4)
        self.one_rdm_l2_pure = build_local_tensor(
            self.one_rdm_full, [self.l2_idx]*2)
        self.two_rdm_l2_pure = build_local_tensor(
            self.two_rdm_full, [self.l2_idx]*4)

        self.one_rdm_ct1 = build_local_tensor(
            self.one_rdm_full, [self.l1_idx, self.l2_idx])
        self.one_rdm_ct2 = build_local_tensor(
            self.one_rdm_full, [self.l2_idx, self.l1_idx])

        self.two_rdm_ct11 = build_local_tensor(
            self.two_rdm_reord_full, [self.l1_idx, self.l1_idx, self.l1_idx, self.l2_idx])
        self.two_rdm_ct12 = build_local_tensor(
            self.two_rdm_reord_full, [self.l1_idx, self.l1_idx, self.l2_idx, self.l1_idx])
        self.two_rdm_ct21 = build_local_tensor(
            self.two_rdm_reord_full, [self.l2_idx, self.l2_idx, self.l2_idx, self.l1_idx])
        self.two_rdm_ct22 = build_local_tensor(
            self.two_rdm_reord_full, [self.l2_idx, self.l2_idx, self.l1_idx, self.l2_idx])

        self.h_eff_ct1 = h_eff(
            build_local_tensor(self.c1, [self.l1_idx, self.l2_idx]),
            build_local_tensor(self.c2, [self.l1_idx, self.l1_idx, self.l1_idx, self.l2_idx]))
        self.h_eff_ct2 = h_eff(
            build_local_tensor(self.c1, [self.l2_idx, self.l1_idx]),
            build_local_tensor(self.c2, [self.l2_idx, self.l2_idx, self.l2_idx, self.l1_idx]))
        self.c2_ct1 = build_local_tensor(
            self.c2, [self.l1_idx, self.l1_idx, self.l1_idx, self.l2_idx])
        self.c2_ct2 = build_local_tensor(
            self.c2, [self.l2_idx, self.l2_idx, self.l2_idx, self.l1_idx])

        self.two_rdm_2ct1 = build_local_tensor(
            self.two_rdm_reord_full, [self.l1_idx, self.l2_idx, self.l1_idx, self.l2_idx])
        self.two_rdm_2ct2 = build_local_tensor(
            self.two_rdm_reord_full, [self.l2_idx, self.l1_idx, self.l2_idx, self.l1_idx])

        self.c2_2ct1 = build_local_tensor(
            self.c2, [self.l1_idx, self.l2_idx, self.l1_idx, self.l2_idx])
        self.c2_2ct2 = build_local_tensor(
            self.c2, [self.l2_idx, self.l1_idx, self.l2_idx, self.l1_idx])

        self.tt = None

    def get_tt_rdms(self):
        if self.tt is None:
            (self.one_rdm_a, self.one_rdm_b), (
                self.two_rdm_aa, self.two_rdm_ab, self.two_rdm_bb) = make_rdm12s(
                    self.fcivec, self.ncas, self.nelecas, reorder=False)
            self.two_rdm_ba = self.two_rdm_ab.transpose(2, 3, 0, 1)
            tt_0 = .5 * (self.two_rdm_aa - self.two_rdm_ab -
                         self.two_rdm_ba + self.two_rdm_bb)
            tt_p = self.two_rdm_ab.transpose(0, 3, 2, 1) - np.einsum(
                'rs, pq->psrq', np.eye(self.ncas), self.one_rdm_a)
            tt_m = self.two_rdm_ba.transpose(0, 3, 2, 1) - np.einsum(
                'rs, pq->psrq', np.eye(self.ncas), self.one_rdm_b)
            self.tt = build_local_tensor(tt_0 - tt_p - tt_m,
                                         [self.l1_idx, self.l1_idx, self.l2_idx, self.l2_idx])
        return self.tt

    def local_ham_est(self, fragment):
        if fragment == 'l1':
            return sum((
                np.einsum('pq, qp', self.c1_pure_l1, self.one_rdm_l1_pure),
                .5 * np.einsum('pqrs, pqrs', self.c2_pure_l1,
                               self.two_rdm_l1_pure)
            ))
        elif fragment == 'l2':
            return sum((
                np.einsum('pq, qp', self.c1_pure_l2, self.one_rdm_l2_pure),
                .5 * np.einsum('pqrs, pqrs', self.c2_pure_l2,
                               self.two_rdm_l2_pure)
            ))

    def h0_disp_est(self):
        h0_disp = np.einsum('pqrs, pq, rs', self.exchange,
                            self.one_rdm_l1_pure, self.one_rdm_l2_pure)
        return h0_disp

    def h_prime_disp_est(self):
        h0_disp = self.h0_disp_est()
        h_disp = np.einsum('pqrs,pqrs', self.exchange, build_local_tensor(
            self.two_rdm_reord_full, [self.l1_idx, self.l1_idx, self.l2_idx, self.l2_idx]))
        return h_disp - h0_disp

    def h_ct_est(self):
        return (
            np.einsum('pq, pq', self.h_eff_ct1, self.one_rdm_ct1),
            np.einsum('pq, pq', self.h_eff_ct2, self.one_rdm_ct2),
            np.einsum('pqrs, pqrs', self.c2_ct1, self.two_rdm_ct11),
            np.einsum('pqrs, pqsr', self.c2_ct1, self.two_rdm_ct12),
            np.einsum('pqrs, pqrs', self.c2_ct2, self.two_rdm_ct21),
            np.einsum('pqrs, pqsr', self.c2_ct2, self.two_rdm_ct22),
        )

    def h_2ct_est(self):
        return (
            .5 * np.einsum('pqrs, pqrs', self.c2_2ct1, self.two_rdm_2ct1),
            .5 * np.einsum('pqrs, pqrs', self.c2_2ct2, self.two_rdm_2ct2)
        )

    def h_tt_est(self):
        self.get_tt_rdms()
        return - np.einsum('psrq, pqrs', build_local_tensor(
            self.c2, [self.l1_idx, self.l2_idx, self.l2_idx, self.l1_idx]), self.tt)


if __name__ == '__main__':
    from fragpt2 import unpack_pyscf, get_expectation, pyscf_ci_to_psi
    from cirq import dirac_notation
    basis = 'unc-cc-pvdz'
    symmetry = 0
    cart = True

    mol = gto.Mole(atom='../../examples/water_ammonia/water-ammonia.xyz',
                   basis=basis, symmetry=symmetry, cart=cart)
    mol.build()
    mf = scf.RHF(mol)
    mf.run()

    nao, mo_energies, mo_coeff = unpack_pyscf(
        "../../examples/water_ammonia/ribo.pyscf")
    # Water ammonia:
    # # for CAS(2,2)(2,2):
    # act_idx_l1_base1 = [4, 11]
    # act_idx_l2_base1 = [8, 27]

    # # for CAS(4,4)(4,4):
    act_idx_l1_base1 = [4, 5, 11, 12]
    act_idx_l2_base1 = [8, 9, 27, 28]

    # # # for CAS(2,2)(4,4):
    # act_idx_l1_base1 = [5, 11]
    # act_idx_l2_base1 = [8, 9, 27, 28]

    # # # for CAS(4,3)(4,4):
    # act_idx_l1_base1 = [4, 5, 11]
    # act_idx_l2_base1 = [8, 9, 27, 28]

    act_idx_l1 = np.array(act_idx_l1_base1) - 1
    act_idx_l2 = np.array(act_idx_l2_base1) - 1
    act_idx = np.concatenate((act_idx_l1, act_idx_l2))

    # SET HERE THE SIZE OF THE ACTIVE SPACE
    ncas = len(act_idx)
    nelecas = ncas - 2 * (ncas // 2 % 2)
    nelec_l1 = nelecas // 2
    nelec_l2 = nelecas // 2
    ncas_l1 = len(act_idx_l1)
    ncas_l2 = len(act_idx_l2)

    print(f'Full: CAS({nelecas},{ncas})')
    print(f'l1: CAS({nelec_l1},{ncas_l1})')
    print(f'l2: CAS({nelec_l2},{ncas_l2})')

    occ_idx = [x for x in range(mol.nelectron//2) if x not in act_idx]

    nroots = 1

    casci = FCI_terms(mf, occ_idx, act_idx,
                      (('l1', ncas_l1, nelec_l1), ('l2', ncas_l2, nelec_l2)),
                      mo_energies=mo_energies, mo_coeff=mo_coeff)

    e_corr_naive = casci.e_l1 + casci.e_l2 - 2 * mf.e_tot
    e_naive = casci.e_l1 + casci.e_l2 - mf.e_tot
    e_tot = casci.e_tot

    print('\n\n')
    print(f'Naive energy:        {e_naive:.6f}')
    print(f'scf energy:          {e_tot:.6f}')
    # print(f'exact energy:        {casci.e_full:.6f}')

    check2_e = casci.c0 + casci.nuclear_repulsion + casci.eci_full

    print(f'Check fci    full:   {check2_e:.6f}')
    print('Corresponding state:', dirac_notation(pyscf_ci_to_psi(casci.fcivec,
                                                                 ncas, nelecas)))

    print(f'fci full CI energy:  {casci.eci_full:.6f}')
    print(
        f'scf CI energy:       {e_tot - casci.c0 - casci.nuclear_repulsion:.6f}')

    h_l1_est = casci.local_ham_est('l1')
    h_l2_est = casci.local_ham_est('l2')
    h0_disp = casci.h0_disp_est()
    h_prime_disp = casci.h_prime_disp_est()
    h_ct_all = casci.h_ct_est()
    h_2ct_all = casci.h_2ct_est()
    h_tt = casci.h_tt_est()

    print(f'Local ham expval l1:   {h_l1_est:.6f}')
    print(f'Local ham expval l2:   {h_l2_est:.6f}')
    print(f'h0 disp expectation:   {h0_disp:.6f}')
    print(f'total h0 expectation:  {h_l1_est + h_l2_est + h0_disp:.6f}')
    print(f'h disp expectation:    {h_prime_disp:.6f}')
    print(
        f'total h disp expval:   {h_l1_est + h_l2_est + h0_disp + h_prime_disp:.6f}')
    print('contributions of h_ct:', *h_ct_all)
    print(f'Total single ct:       {sum(h_ct_all):.6f}')
    print(
        f'total h expval:        {h_l1_est + h_l2_est + h0_disp + h_prime_disp + sum(h_ct_all):.6f}')
    print('contributions of h_2ct:', *h_2ct_all)
    print(f'Total double ct:       {sum(h_2ct_all):.6f}')
    print(
        f'h SS expval:           {h_l1_est + h_l2_est + h0_disp + h_prime_disp + sum(h_ct_all) + sum(h_2ct_all):.6f}')
    print(
        f'Total TT terms:        {h_tt:.6f}')
    h_tot = h_l1_est + h_l2_est + h0_disp + \
        h_prime_disp + sum(h_ct_all) + sum(h_2ct_all) + h_tt
    print(
        f'FINAL h expval:        {h_tot:.6f}')

    e_pt2 = casci.run_fragpt2()
    # e_pt3, e_pt2 = casci.run_fragpt3()

    print()
    # print(f'embedding energy w pt3:  {e_tot + e_pt2 + e_pt3:.6f}')
    print(f'embedding energy wo pt3: {e_tot + e_pt2:.6f}')
    print(f'Pt2 correction:          {e_pt2:.8f}')

    e_ct1_ci = casci.e_ct1 - mf.e_tot
    e_ct2_ci = casci.e_ct2 - mf.e_tot

    print(f'CT1 CI energy:       {e_ct1_ci:.6f}')
    print(f'CT2 CI energy:       {e_ct2_ci:.6f}')

    e_pt2_ct1, e_pt2_ct2 = casci.run_fragpt2_ct()

    print(f'CT1 PT2 energy:      {e_pt2_ct1:.6f}')
    print(f'CT2 PT2 energy:      {e_pt2_ct2:.6f}')

    print(f'2 * CT PT2 energy:   {2*(e_pt2_ct1 + e_pt2_ct2):.6f}')
    print(
        f'Full embedding E:    {e_tot + e_pt2 + 2*(e_pt2_ct1 + e_pt2_ct2):.6f}')
