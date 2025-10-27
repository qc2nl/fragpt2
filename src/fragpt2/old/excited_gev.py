#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 17:57:39 2023

@author: emielkoridon
"""


import numpy as np
import scipy
from pyscf import scf, mcscf, ao2mo
from pyscf.fci.direct_spin0 import trans_rdm12, trans_rdm1s
from pyscf.fci.addons import overlap
from fragpt2.active_space import molecular_hamiltonian_coefficients


class Embed_excited():
    """Class for embedding two CASCI calculations on fragmented MOs."""

    def __init__(self, mf: scf.hf.RHF,
                 occ_idx, act_idx,
                 fragment_cas,
                 mo_energies=None, mo_coeff=None, verbose=0):
        """
        Initialize CASCI embedding class.

        Args:
            mf (scf.hf.RHF): Full RHF object from pyscf
            occ_idx (list): Occupied indices that are always kept doubly occupied
            act_idx (list): Active indices ordered by fragment, then occupied - virtual
            fragment_cas (tuple): Tuple (N,M,L,K) of ncas, nelecas for the fragments
                respectively
            mo_energies (np.ndarray, optional): MO energies
            mo_coeff (np.ndarray, optional): MO coefficients

        """
        if mo_energies is not None:
            mf.mo_energy = mo_energies
        if mo_coeff is not None:
            mf.mo_coeff = mo_coeff

        froc = mf.get_occ(mo_energy=mf.mo_energy)
        mf.mo_occ = froc

        # copy RHF object
        self.mf_full = mf
        self.mf_l1 = mf
        self.mf_l2 = mf

        # Set all fragment information
        self.occ_idx = occ_idx
        self.act_idx = act_idx

        self.nao = mf.mol.nao
        self.nmo = len(mf.mo_energy)
        self.ncas = len(act_idx)
        self.ncore = len(occ_idx) * 2

        self.ncas_l1, self.nelec_l1, self.ncas_l2, self.nelec_l2 = fragment_cas

        occ_idx_l1 = list(range(self.nelec_l1//2))
        virt_idx_l1 = list(range(self.nelec_l1//2, self.ncas_l1))

        occ_idx_l2 = list(range(self.ncas_l1, self.ncas_l1 + self.nelec_l2//2))
        virt_idx_l2 = list(
            range(self.ncas_l1 + self.nelec_l2//2, self.ncas_l1 + self.ncas_l2))

        self.l1_idx = occ_idx_l1 + virt_idx_l1
        self.act_idx_l1 = np.array(act_idx)[self.l1_idx]
        self.l2_idx = occ_idx_l2 + virt_idx_l2
        self.act_idx_l2 = np.array(act_idx)[self.l2_idx]

        self.nelecas = self.nelec_l1 + self.nelec_l2

        # Set MO integrals for Hamiltonian
        int1e_ao = mf.mol.intor('int1e_kin') + mf.mol.intor('int1e_nuc')

        self.nuclear_repulsion = mf.mol.energy_nuc()
        self.one_body_integrals = mf.mo_coeff.T @ int1e_ao @ mf.mo_coeff
        self.two_body_integrals = ao2mo.kernel(
            mf.mol, mf.mo_coeff, aosym=1).reshape(*[self.nmo]*4)

        # Run fragment CASCIs:
        self.cas_full = None
        self.cas_l1 = None
        self.cas_l2 = None

        self.verbose = verbose

    def run(self, nroots=1):
        """Run full embedding algorithm by constructing the Hamiltonian and overlap matrix
        and solving the generalized eigenvalue problem"""
        hamiltonian, overlap = self.construct_embed_hamiltonian_ovlp(
            nroots=nroots)
        return scipy.linalg.eigh(hamiltonian, b=overlap, lower=False)

    def run_full_casci(self, nroots=1, fix_singlet=True):
        """Run CASCI using PySCF on the full active space"""
        if self.cas_full is None or self.cas_full.fcisolver.nroots < nroots:
            self.cas_full = mcscf.CASCI(self.mf_full, self.ncas, self.nelecas)
            self.cas_full.verbose = self.verbose
            self.cas_full.canonicalization = False
            mo_full = self.cas_full.sort_mo(self.act_idx, base=0)
            self.cas_full.fcisolver.nroots = nroots
            if fix_singlet:
                self.cas_full.fix_spin_(shift=1.5, ss=0)
            self.e_full, e_ci_full, self.civec_full, _, _ = self.cas_full.kernel(
                mo_full)

            print('Performed full CASCI calculation. energy:', self.e_full)

    def run_l1_casci(self, nroots=1, fix_singlet=True):
        """Run CASCI on the first fragment"""
        if self.cas_l1 is None or self.cas_l1.fcisolver.nroots < nroots:
            self.cas_l1 = mcscf.CASCI(self.mf_l1, self.ncas_l1, self.nelec_l1)
            self.cas_l1.verbose = self.verbose
            self.cas_l1.canonicalization = False
            mo_l1 = self.cas_l1.sort_mo(self.act_idx_l1, base=0)
            self.cas_l1.fcisolver.nroots = nroots
            if fix_singlet:
                self.cas_l1.fix_spin_(shift=1.5, ss=0)
            self.e_l1, e_ci_l1, self.civec_l1, _, _ = self.cas_l1.kernel(mo_l1)

            print('Performed l1 CASCI calculation. energy:', self.e_l1)

    def run_l2_casci(self, nroots=1, fix_singlet=True):
        """Run CASCI on the second fragment"""
        if self.cas_l2 is None or self.cas_l2.fcisolver.nroots < nroots:
            self.cas_l2 = mcscf.CASCI(self.mf_l2, self.ncas_l2, self.nelec_l2)
            self.cas_l2.verbose = self.verbose
            self.cas_l2.canonicalization = False
            mo_l2 = self.cas_l2.sort_mo(self.act_idx_l2, base=0)
            self.cas_l2.fcisolver.nroots = nroots
            if fix_singlet:
                self.cas_l2.fix_spin_(shift=1.5, ss=0)
            self.e_l2, e_ci_l2, self.civec_l2, _, _ = self.cas_l2.kernel(mo_l2)

            print('Performed l2 CASCI calculation. energy:', self.e_l2)

    def build_d_p_mat_l1_l2(self, civec_l1_bra, civec_l2_bra, civec_l1_ket, civec_l2_ket):
        r"""
        Constructs the one and two transition density matrices. They are defined as follows:

        .. math::
            D_{tu}^{A/B} = \langle\Psi_{A_1}|\langle\Psi_{B_1}| E_{tu}
            |\Psi_{A_2}\rangle|\Psi_{B_2}\rangle\\
            P_{tuvw}^{A/B} = \langle\Psi_{A_1}|\langle\Psi_{B_1}| e_{tuvw}
            |\Psi_{A_2}\rangle|\Psi_{B_2}\rangle \\
            = \langle\Psi_{A_1}|\langle\Psi_{B_1}| E_{tu} E_{vw}
            |\Psi_{A_2}\rangle|\Psi_{B_2}\rangle - \delta_{qr} D_{tu}^{A/B}

        This function assumes there is no entanglement between system A and B, thus the
        expressions of the TDMs factorize. See Eq. 50 and 51 in the overleaf.

        Args:
            civec_l1_bra (pyscf.fci.direct_spin1.FCIvector):
                FCIvector object representing :math:`|\Psi_{A_1}\rangle`.
            civec_l2_bra (pyscf.fci.direct_spin1.FCIvector):
                FCIvector object representing :math:`|\Psi_{B_1}\rangle`.
            civec_l1_ket (pyscf.fci.direct_spin1.FCIvector):
                FCIvector object representing :math:`|\Psi_{A_2}\rangle`.
            civec_l2_ket (pyscf.fci.direct_spin1.FCIvector):
                FCIvector object representing :math:`|\Psi_{B_2}\rangle`.

        Returns:
            d_mat (np.ndarray):
                one-body transition density matrix :math:`D_{tu}^{A/B}`.
            p_mat (np.ndarray):
                two-body transition density matrix :math:`P_{tuvw}^{A/B}`.

        """

        # Construct overlaps on respective fragments
        ci_l1_ovlp = overlap(civec_l1_bra, civec_l1_ket,
                             self.ncas_l1, self.nelec_l1)
        ci_l2_ovlp = overlap(civec_l2_bra, civec_l2_ket,
                             self.ncas_l2, self.nelec_l2)

        # Construct all the trans 1 and 2-RDMs on respective fragments
        ci_l1_singles, ci_l1_doubles = trans_rdm12(
            civec_l1_bra, civec_l1_ket, self.ncas_l1, self.nelec_l1, reorder=False)
        ci_l2_singles, ci_l2_doubles = trans_rdm12(
            civec_l2_bra, civec_l2_ket, self.ncas_l2, self.nelec_l2, reorder=False)

        # Construct all the unrestricted trans 1-RDMs on respective fragments
        ci_l1_singles_a, ci_l1_singles_b = trans_rdm1s(
            civec_l1_bra, civec_l1_ket, self.ncas_l1, self.nelec_l1)
        ci_l2_singles_a, ci_l2_singles_b = trans_rdm1s(
            civec_l2_bra, civec_l2_ket, self.ncas_l2, self.nelec_l2)

        d_mat = np.zeros((self.ncas, self.ncas))

        # Eq. 50
        d_mat[np.ix_(self.l1_idx, self.l1_idx)] = ci_l1_singles * ci_l2_ovlp
        d_mat[np.ix_(self.l2_idx, self.l2_idx)] = ci_l2_singles * ci_l1_ovlp

        p_mat = np.zeros((self.ncas, self.ncas, self.ncas, self.ncas))

        # Eq. 51a
        p_mat[np.ix_(*[self.l1_idx]*4)] = ci_l1_doubles * ci_l2_ovlp

        # Eq. 51b
        p_mat[np.ix_(*[self.l2_idx]*4)] = ci_l2_doubles * ci_l1_ovlp

        # Eq. 51c
        p_mat[np.ix_(self.l1_idx, self.l1_idx, self.l2_idx, self.l2_idx)] = np.einsum(
            'tu, vw -> tuvw', ci_l1_singles, ci_l2_singles)

        # Eq. 51d
        p_mat[np.ix_(self.l2_idx, self.l2_idx, self.l1_idx, self.l1_idx)] = np.einsum(
            'vw, tu -> tuvw', ci_l1_singles, ci_l2_singles)

        # Eq. 51e
        p_mat[np.ix_(self.l1_idx, self.l2_idx, self.l2_idx, self.l1_idx)] = - np.einsum(
            'tw, vu -> tuvw', ci_l1_singles_a, ci_l2_singles_a) - np.einsum(
                'tw, vu -> tuvw', ci_l1_singles_b, ci_l2_singles_b) + \
            np.einsum('tw, uv->tuvw',
                      d_mat[np.ix_(self.l1_idx, self.l1_idx)],
                      np.eye(self.ncas)[np.ix_(self.l2_idx, self.l2_idx)])

        # Eq. 51f
        p_mat[np.ix_(self.l2_idx, self.l1_idx, self.l1_idx, self.l2_idx)] = - np.einsum(
            'vu, tw -> tuvw', ci_l1_singles_a, ci_l2_singles_a) - np.einsum(
                'vu, tw -> tuvw', ci_l1_singles_b, ci_l2_singles_b) + \
            np.einsum('tw, uv->tuvw',
                      d_mat[np.ix_(self.l2_idx, self.l2_idx)],
                      np.eye(self.ncas)[np.ix_(self.l1_idx, self.l1_idx)])

        # Perform reordering according to anti-commutation relation
        p_mat = p_mat - np.einsum('uv, tw->tuvw', np.eye(self.ncas), d_mat)

        return d_mat, p_mat

    def get_active_integrals(self, with_nuc=False):
        """Get active space integrals (zero, one and two-body). See documentation of
        `molecular_hamiltonian_coefficients`."""
        if with_nuc:
            return molecular_hamiltonian_coefficients(
                self.nuclear_repulsion, self.one_body_integrals, self.two_body_integrals,
                self.occ_idx, self.act_idx)
        else:
            return molecular_hamiltonian_coefficients(
                0, self.one_body_integrals, self.two_body_integrals,
                self.occ_idx, self.act_idx)

    def construct_embed_hamiltonian_ovlp(self, nroots=1):
        """
        Construct embedding Hamiltonian and overlap matrix in the basis of all possible
        combinations of states on each fragment. This results in (nroots+1)**2 x (nroots+1)**2
        matrix size.

        Args:
            nroots (int, optional): Number of states considered on each fragment on top of the
                Hartree-Fock state. Defaults to 1.

        Returns:
            hamiltonian (np.ndarray): Hamiltonian matrix of size (nroots+1)**2 x (nroots+1)**2.
            ovlp (np.ndarray): Overlap matrix of size (nroots+1)**2 x (nroots+1)**2.

        """
        self.run_l1_casci(nroots)
        self.run_l2_casci(nroots)

        # Construct active space integrals to contract with the (transition) RDMs
        c0, c1, c2 = self.get_active_integrals()

        # Get ci vectors from the fragment calculations and put them in a list
        if type(self.civec_l1) == list:
            civec_l1_hf = np.zeros_like(self.civec_l1[0])
        else:
            civec_l1_hf = np.zeros_like(self.civec_l1)
        civec_l1_hf[0, 0] = 1.

        if type(self.civec_l2) == list:
            civec_l2_hf = np.zeros_like(self.civec_l2[0])
        else:
            civec_l2_hf = np.zeros_like(self.civec_l2)
        civec_l2_hf[0, 0] = 1.

        if type(self.civec_l1) == list:
            civec_tot_l1 = [civec_l1_hf] + self.civec_l1[:nroots]
        else:
            civec_tot_l1 = [civec_l1_hf] + [self.civec_l1]
        if type(self.civec_l2) == list:
            civec_tot_l2 = [civec_l2_hf] + self.civec_l2[:nroots]
        else:
            civec_tot_l2 = [civec_l2_hf] + [self.civec_l2]

        hamiltonian = np.zeros(((nroots+1)**2, (nroots+1)**2))
        ovlp = np.eye((nroots+1)**2)

        # Build the diagonal of the Hamiltonian
        k = 0
        for i, civec_l1_braket in enumerate(civec_tot_l1):
            for j, civec_l2_braket in enumerate(civec_tot_l2):
                d_mat_l1_l2, p_mat_l1_l2 = self.build_d_p_mat_l1_l2(
                    civec_l1_braket, civec_l2_braket,
                    civec_l1_braket, civec_l2_braket)
                e_l1_l2 = sum((
                    c0,
                    np.einsum('pq, pq', c1, d_mat_l1_l2),
                    np.einsum('pqrs, pqrs', c2, p_mat_l1_l2)))
                hamiltonian[k, k] = e_l1_l2
                k += 1

        # Build all possible basis states
        civecs = []
        for i in range(nroots+1):
            for j in range(nroots+1):
                civecs.append((civec_tot_l1[i], civec_tot_l2[j]))

        # Compute upper right triangle of Hamiltonian and overlap matrix
        for i, (civec_l1_bra, civec_l2_bra) in enumerate(civecs):
            for j, (civec_l1_ket, civec_l2_ket) in enumerate(civecs):
                if j > i:
                    ci_l1_ovlp = overlap(
                        civec_l1_bra, civec_l1_ket, self.ncas_l1, self.nelec_l1)
                    ci_l2_ovlp = overlap(
                        civec_l2_bra, civec_l2_ket, self.ncas_l2, self.nelec_l2)
                    d_mat, p_mat = self.build_d_p_mat_l1_l2(
                        civec_l1_bra, civec_l2_bra, civec_l1_ket, civec_l2_ket)
                    mat_el = sum((
                        c0 * ci_l1_ovlp * ci_l2_ovlp,
                        np.einsum('pq, pq', c1, d_mat),
                        np.einsum('pqrs, pqrs', c2, p_mat)))

                    hamiltonian[i, j] = mat_el
                    ovlp[i, j] = ci_l1_ovlp * ci_l2_ovlp

        return hamiltonian, ovlp
