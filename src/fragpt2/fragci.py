#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 11:42:03 2024

@author: emielkoridon
"""

import numpy as np
from pyscf import gto, fci, ao2mo

from fragpt2.utils.active_space import get_active_integrals
from pyscf.fci.direct_spin0 import make_rdm1, make_rdm12


def run_fci(c1, c2, norb, nelec, nroots=1, fix_singlet=True, verbose=0):
    """
    Run FCI using PySCF on specified integrals

    Args:
        c1 (np.ndarray): 1-electron integrals.

        c2 (np.ndarray): 2-electron integrals in chemists convention.

        norb (int): Number of orbitals.

        nelec (int): Number of electrons.

        nroots (int, optional): Number of eigenstates to compute. Defaults to 1.

        fix_singlet (bool, optional): Constrain to singlet spin sector. Defaults to True.

        verbose (int, optional): Verbose level. Defaults to 0.

    Returns:
        e_ci (float): CI energy.

        civec (np.ndarray): CI vector.

    """
    mc = fci.direct_spin1.FCI()
    mc.max_cycle = 100
    mc.conv_tol = 1e-8
    mc.verbose = verbose
    mc.canonicalization = False
    mc.nroots = nroots
    if fix_singlet:
        mc = fci.addons.fix_spin_(mc, ss=0)
    return mc.kernel(c1, c2, norb, nelec)


def init_hf_dm1(norb, nelec):
    """Initialize 1-particle reduced density matrix of a HF state"""
    dm1 = np.zeros((norb, norb))
    np.fill_diagonal(dm1, [2. for x in range(nelec // 2)] + [
        0. for x in range(norb - nelec // 2)])
    return dm1


def init_hf_dm2(norb, nelec):
    """Initialize 2-particle reduced density matrix of a HF state"""
    dm1 = init_hf_dm1(norb, nelec)
    dm2 = np.einsum('ij, kl->ijkl', dm1, dm1) - .5 * np.einsum(
        'il, kj->ijkl', dm1, dm1)
    return dm2


def get_exchange(indices, integrals):
    _, c2 = integrals
    l1_idx, l2_idx = indices
    return (c2 - .5 * c2.transpose(0, 3, 2, 1))[
        np.ix_(l1_idx, l1_idx, l2_idx, l2_idx)]


class FragCI:
    """Class for embedding self-consistent fragments."""

    def __init__(self, mol: gto.mole.Mole,
                 occ_idx, act_idx,
                 fragments,
                 mo_energies, mo_coeff, verbose=0):
        """
        Initialize CASCI embedding class.

        Args:
            mol (gto.mole.Mole): Full RHF object from pyscf

            occ_idx (list): Occupied indices that are always kept doubly occupied

            act_idx (list): Active indices ordered by fragment, then occupied - virtual

            fragments (tuple): Tuple (ncas,nelecas) for each fragment

            mo_energies (np.ndarray): MO energies

            mo_coeff (np.ndarray): MO coefficients

            verbose (Bool, optional): Control verbosity

        """
        # Set fcisolver. This can be changed to any other method
        # that outputs CI energy and CI vector in PySCF convention.
        self.fcisolver = run_fci

        # ---- Complete Active Space indices ----
        self.occ_idx = occ_idx
        self.act_idx = act_idx
        self.nao = mol.nao
        self.nmo = len(mo_energies)
        self.ncas = len(act_idx)
        self.ncore = len(occ_idx) * 2

        # ---- Fragment information ----

        # Set active space sizes
        ((self.ncas_l1, self.nelec_l1),
         (self.ncas_l2, self.nelec_l2)) = fragments

        # Set active space indices (also of CT spaces)
        l1_idx_occ = list(range(self.nelec_l1//2))
        l1_idx_virt = list(range(self.nelec_l1//2, self.ncas_l1))
        l2_idx_occ = list(range(self.ncas_l1, self.ncas_l1 + self.nelec_l2//2))
        l2_idx_virt = list(
            range(self.ncas_l1 + self.nelec_l2//2, self.ncas_l1 + self.ncas_l2))
        self.l1_idx = l1_idx_occ + l1_idx_virt
        self.act_idx_l1 = np.array(act_idx)[self.l1_idx]
        self.l2_idx = l2_idx_occ + l2_idx_virt
        self.act_idx_l2 = np.array(act_idx)[self.l2_idx]
        self.ct1_idx = l1_idx_occ + l2_idx_virt
        self.act_idx_ct1 = np.array(act_idx)[self.ct1_idx]
        self.ct2_idx = l2_idx_occ + l1_idx_virt
        self.act_idx_ct2 = np.array(act_idx)[self.ct2_idx]
        self.ncas_ct1 = len(self.act_idx_ct1)
        self.ncas_ct2 = len(self.act_idx_ct2)
        self.occ_idx_l1 = np.append(np.array(occ_idx), act_idx[l2_idx_occ])
        self.occ_idx_l2 = np.append(np.array(occ_idx), act_idx[l1_idx_occ])
        self.nelecas = self.nelec_l1 + self.nelec_l2

        # ---- Active space integrals ----
        int1e_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        self.nuclear_repulsion = mol.energy_nuc()
        self.one_body_integrals = mo_coeff.T @ int1e_ao @ mo_coeff
        self.two_body_integrals = ao2mo.kernel(
            mol, mo_coeff, aosym=1).reshape(*[self.nmo]*4)
        self.c0, self.c1, self.c2 = get_active_integrals(
            self.one_body_integrals, self.two_body_integrals, occ_idx, act_idx)

        # Compute coulomb - exchange integrals
        self.exchange = get_exchange(
            (self.l1_idx, self.l2_idx), (self.c1, self.c2))

        # Set verbosity
        self.verbose = verbose

        # Set initial 1-RDMs to HF
        self.one_rdm_l1 = init_hf_dm1(self.ncas_l1, self.nelec_l1)
        self.one_rdm_l2 = init_hf_dm1(self.ncas_l2, self.nelec_l2)
        self.two_rdm_l1_ord = init_hf_dm1(self.ncas_l1, self.nelec_l1)
        self.two_rdm_l2_ord = init_hf_dm1(self.ncas_l2, self.nelec_l2)

    def run_full_casci(self, nroots=1, fix_singlet=True, verbose=0):
        """Run CASCI using PySCF on the full active space"""
        self.e_ci_full, self.civec_full = self.fcisolver(
            self.c1, self.c2, self.ncas, self.nelecas, nroots, fix_singlet, verbose)
        self.e_full = self.e_ci_full + self.c0 + self.nuclear_repulsion
        if self.verbose:
            print('Performed full CASCI calculation. energy:', self.e_full)

    def run_l1_casci(self, nroots=1, fix_singlet=True, verbose=0):
        """Run CASCI on the first fragment"""
        c0, c1, c2 = get_active_integrals(
            self.one_body_integrals, self.two_body_integrals, self.occ_idx_l1, self.act_idx_l1)
        self.e_ci_l1, self.civec_l1_cas = self.fcisolver(
            c1, c2, self.ncas_l1, self.nelec_l1, nroots, fix_singlet, verbose)
        self.e_l1 = self.e_ci_l1 + c0 + self.nuclear_repulsion
        if self.verbose:
            print('Performed l1 CASCI calculation. energy:  ', self.e_l1)

    def run_l2_casci(self, nroots=1, fix_singlet=True, verbose=0):
        """Run CASCI on the second fragment"""
        c0, c1, c2 = get_active_integrals(
            self.one_body_integrals, self.two_body_integrals, self.occ_idx_l2, self.act_idx_l2)
        self.e_ci_l2, self.civec_l2_cas = self.fcisolver(
            c1, c2, self.ncas_l2, self.nelec_l2, nroots, fix_singlet, verbose)
        self.e_l2 = self.e_ci_l2 + c0 + self.nuclear_repulsion
        if self.verbose:
            print('Performed l2 CASCI calculation. energy:  ', self.e_l2)

    def run_ct1_casci(self, nroots=1, fix_singlet=True, verbose=0):
        """Run CASCI on the CT1 space"""
        c0, c1, c2 = get_active_integrals(
            self.one_body_integrals, self.two_body_integrals, self.occ_idx_l1, self.act_idx_ct1)
        self.e_ci_ct1, self.civec_ct1_cas = self.fcisolver(
            c1, c2, self.ncas_ct1, self.nelec_l1, nroots, fix_singlet, verbose)
        self.e_ct1 = self.e_ci_ct1 + c0 + self.nuclear_repulsion
        if self.verbose:
            print('Performed ct1 CASCI calculation. energy:  ', self.e_ct1)

    def run_ct2_casci(self, nroots=1, fix_singlet=True, verbose=0):
        """Run CASCI on the CT2 space"""
        c0, c1, c2 = get_active_integrals(
            self.one_body_integrals, self.two_body_integrals, self.occ_idx_l2, self.act_idx_ct2)
        self.e_ci_ct2, self.civec_ct2_cas = self.fcisolver(
            c1, c2, self.ncas_ct2, self.nelec_l2, nroots, fix_singlet, verbose)
        self.e_ct2 = self.e_ci_ct2 + c0 + self.nuclear_repulsion
        if self.verbose:
            print('Performed ct2 CASCI calculation. energy:  ', self.e_ct2)

    def run_mean_field(self, verbose=0):
        one_rdm_l1 = init_hf_dm1(self.ncas_l1, self.nelec_l1)
        one_rdm_l2 = init_hf_dm1(self.ncas_l2, self.nelec_l2)
        two_rdm_l1_ord = init_hf_dm2(self.ncas_l1, self.nelec_l1)
        two_rdm_l2_ord = init_hf_dm2(self.ncas_l2, self.nelec_l2)

        self.e_mf = self.h0_expval_rdms(one_rdm_l1, two_rdm_l1_ord,
                                        one_rdm_l2, two_rdm_l2_ord) + self.nuclear_repulsion

    def run_self_consistent_casci(self, fix_singles=True, e_conv=1e-10):
        e_old = de = 1e99

        n = 0
        while np.abs(de) > e_conv:
            if n == 0:
                tmp1 = np.copy(self.civec_l1_cas)
                tmp2 = np.copy(self.civec_l2_cas)
                civec_hf_l1 = np.zeros_like(tmp1)
                civec_hf_l1[0, 0] = 1.
                civec_hf_l2 = np.zeros_like(tmp2)
                civec_hf_l2[0, 0] = 1.

                ce1 = self.h0_expval_civec(
                    tmp1, civec_hf_l2) + self.nuclear_repulsion
                ce2 = self.h0_expval_civec(
                    civec_hf_l1, tmp2) + self.nuclear_repulsion

                self.e_naive = self.h0_expval_civec(
                    tmp1, tmp2) + self.nuclear_repulsion
                if self.verbose:
                    print('should be CASCI l1:', ce1)
                    print('should be CASCI l2:', ce2)
                    print('Naive energy is:   ', self.e_naive)

            else:
                e_l2, self.one_rdm_l2, self.civec_l2 = self.get_fragment_energy(
                    'l2')
                e_l1, self.one_rdm_l1, self.civec_l1 = self.get_fragment_energy(
                    'l1')
                e_tot = self.h0_expval_civec(
                    self.civec_l1, self.civec_l2, set_rdms=True)

                de = e_old - e_tot
                e_old = e_tot
                if self.verbose:
                    print('\nIter:', n)
                    print('CI e_l1:', e_l1)
                    print('CI e_l2:', e_l2)
                    print('e_tot', e_tot + self.nuclear_repulsion)
                    print('de:', de)

            n += 1

        if self.verbose:
            print('Performed self-consistent CASCI calculation. energy:',
                  e_tot + self.nuclear_repulsion)

        return e_tot + self.nuclear_repulsion, e_l1, e_l2

    def get_fragment_energy(self, fragment):
        c1_eff, c2_eff = self.get_h1_h2_eff_fragment(fragment)
        if fragment == 'l1':
            e, fcivec = self.fcisolver(
                c1_eff, c2_eff, self.ncas_l1, self.nelec_l1)
            one_rdm_frag = make_rdm1(fcivec, self.ncas_l1, self.nelec_l1).T
        if fragment == 'l2':
            e, fcivec = self.fcisolver(
                c1_eff, c2_eff, self.ncas_l2, self.nelec_l2)
            one_rdm_frag = make_rdm1(fcivec, self.ncas_l2, self.nelec_l2).T
        return e, one_rdm_frag, fcivec

    def get_h1_h2_eff_fragment(self, fragment):
        if fragment == 'l1':
            c2_eff = self.c2[np.ix_(*[self.l1_idx]*4)]
            c1_eff = self.c1[np.ix_(*[self.l1_idx]*2)] + np.einsum(
                'rs, pqrs->pq', self.one_rdm_l2,
                self.exchange)
        elif fragment == 'l2':
            c2_eff = self.c2[np.ix_(*[self.l2_idx]*4)]
            c1_eff = self.c1[np.ix_(*[self.l2_idx]*2)] + np.einsum(
                'pq, pqrs->rs', self.one_rdm_l1,
                self.exchange)
        else:
            raise ValueError(f'No fragment named {fragment}')

        return c1_eff, c2_eff

    def h0_expval_rdms(self, one_rdm_l1, two_rdm_l1_ord, one_rdm_l2, two_rdm_l2_ord, w_core=True):
        c1_eff_l1 = self.c1[np.ix_(*[self.l1_idx]*2)]
        c2_eff_l1 = self.c2[np.ix_(*[self.l1_idx]*4)]

        c1_eff_l2 = self.c1[np.ix_(*[self.l2_idx]*2)]
        c2_eff_l2 = self.c2[np.ix_(*[self.l2_idx]*4)]

        # import pdb
        # pdb.set_trace()

        e_l1 = np.einsum('pq,pq', c1_eff_l1, one_rdm_l1) + .5 * np.einsum(
            'pqrs,pqrs', c2_eff_l1, two_rdm_l1_ord)
        e_l2 = np.einsum('pq,pq', c1_eff_l2, one_rdm_l2) + .5 * np.einsum(
            'pqrs,pqrs', c2_eff_l2, two_rdm_l2_ord)

        c_eff = np.einsum('pqrs,pqrs', self.exchange,
                          np.einsum('pq,rs->pqrs', one_rdm_l1, one_rdm_l2))

        if w_core:
            return self.c0 + e_l1 + e_l2 + c_eff
        else:
            return e_l1 + e_l2 + c_eff

    def h0_expval_civec(self, civec_l1, civec_l2, w_core=True, set_rdms=False):
        if set_rdms:
            self.one_rdm_l1, self.two_rdm_l1_ord = make_rdm12(
                civec_l1, self.ncas_l1, self.nelec_l1)
            self.one_rdm_l2, self.two_rdm_l2_ord = make_rdm12(
                civec_l2, self.ncas_l2, self.nelec_l2)

            return self.h0_expval_rdms(self.one_rdm_l1, self.two_rdm_l1_ord,
                                       self.one_rdm_l2, self.two_rdm_l2_ord,
                                       w_core=w_core)
        else:
            one_rdm_l1, two_rdm_l1_ord = make_rdm12(
                civec_l1, self.ncas_l1, self.nelec_l1)
            one_rdm_l2, two_rdm_l2_ord = make_rdm12(
                civec_l2, self.ncas_l2, self.nelec_l2)

            return self.h0_expval_rdms(
                one_rdm_l1, two_rdm_l1_ord, one_rdm_l2, two_rdm_l2_ord, w_core=w_core)

    def h0_expval(self, w_core=True):
        return self.h0_expval_rdms(self.one_rdm_l1, self.two_rdm_l1_ord,
                                   self.one_rdm_l2, self.two_rdm_l2_ord,
                                   w_core=w_core)


if __name__ == '__main__':
    from pyscf import gto, scf
    from fragpt2 import unpack_pyscf

    basis = 'unc-cc-pvdz'
    symmetry = 0
    cart = True

    mol = gto.Mole(atom='../../examples/water_ammonia/water-ammonia.xyz',
                   basis=basis, symmetry=symmetry, cart=cart)
    mol.max_memory = 8000
    mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol)
    mf.run()

    nao, mo_energies, mo_coeff = unpack_pyscf(
        "../../examples/water_ammonia/ribo.pyscf")

    # for CAS(4,4)(4,4):
    act_idx_l1_base1 = [4, 5, 11, 12]
    act_idx_l2_base1 = [8, 9, 27, 28]

    act_idx_l1 = np.array(act_idx_l1_base1) - 1
    act_idx_l2 = np.array(act_idx_l2_base1) - 1
    act_idx = np.concatenate((act_idx_l1, act_idx_l2))

    # # SET HERE THE SIZE OF THE ACTIVE SPACE
    ncas = len(act_idx)
    nelecas = ncas - 2 * (ncas // 2 % 2)
    nelec_l1 = nelecas // 2
    nelec_l2 = nelecas // 2
    ncas_l1 = ncas // 2
    ncas_l2 = ncas // 2

    print(f'l1: CAS({nelec_l1},{ncas_l1})')
    print(f'l2: CAS({nelec_l2},{ncas_l2})')

    occ_idx = [x for x in range(mol.nelectron//2) if x not in act_idx]

    nroots = 1

    casci = FragCI(mol, occ_idx, act_idx,
                   ((ncas_l1, nelec_l1), (ncas_l2, nelec_l2)),
                   mo_energies, mo_coeff, verbose=1)

    casci.run_mean_field()
    casci.run_full_casci(nroots)
    casci.run_l1_casci(nroots)
    casci.run_l2_casci(nroots)
    casci.run_ct1_casci(nroots)
    casci.run_ct2_casci(nroots)

    e_tot, e_l1, e_l2 = casci.run_self_consistent_casci()

    e_corr_naive = casci.e_l1 + casci.e_l2 - 2 * mf.e_tot
    e_sumcorr = casci.e_l1 + casci.e_l2 - mf.e_tot

    print(f'HF energy of PySCF:      {mf.e_tot:.6f}')
    print(f'HF energy ROSE orbitals: {casci.e_mf:.6f}')
    print(f'Naive energy:            {casci.e_naive:.6f}')
    print(f'Sum of corr energies:    {e_sumcorr:.6f}')
    print(f'scf energy:              {e_tot:.6f}')
    print(f'exact energy:            {casci.e_full:.6f}')
