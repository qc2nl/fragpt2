#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 11:50:47 2024

@author: emielkoridon
"""
import numpy as np
from pyscf import fci


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
    mc = fci.direct_nosym.FCI()
    mc.max_cycle = 100
    mc.conv_tol = 1e-8
    mc.verbose = verbose
    mc.canonicalization = False
    mc.nroots = nroots
    if fix_singlet:
        mc = fci.addons.fix_spin_(mc, ss=0)
    return mc.kernel(c1, c2, norb, nelec)


def get_h0_ints(casci):
    c1 = np.zeros_like(casci.c1)
    c2 = np.zeros_like(casci.c2)

    c1[np.ix_(*[casci.l1_idx]*2)] = casci.c1[np.ix_(*[casci.l1_idx]*2)]
    c1[np.ix_(*[casci.l2_idx]*2)] = casci.c1[np.ix_(*[casci.l2_idx]*2)]

    c1[np.ix_(*[casci.l1_idx]*2)] += np.einsum('pqrs, rs->pq', casci.exchange, casci.one_rdm_l2)
    c1[np.ix_(*[casci.l2_idx]*2)] += np.einsum('pqrs, pq->rs', casci.exchange, casci.one_rdm_l1)

    c_eff = np.einsum('pqrs,pqrs', casci.exchange,
                      np.einsum('pq,rs->pqrs', casci.one_rdm_l1, casci.one_rdm_l2))

    c2[np.ix_(*[casci.l1_idx]*4)] = casci.c2[np.ix_(*[casci.l1_idx]*4)]
    c2[np.ix_(*[casci.l2_idx]*4)] = casci.c2[np.ix_(*[casci.l2_idx]*4)]

    return casci.c0 - c_eff + casci.nuclear_repulsion, c1, c2


def get_hdisp_ints(casci):
    c1 = np.zeros_like(casci.c1)
    c2 = np.zeros_like(casci.c2)

    c1[np.ix_(*[casci.l1_idx]*2)] -= np.einsum('pqrs, rs->pq', casci.exchange, casci.one_rdm_l2)
    c1[np.ix_(*[casci.l2_idx]*2)] -= np.einsum('pqrs, pq->rs', casci.exchange, casci.one_rdm_l1)

    c2[np.ix_(casci.l1_idx, casci.l1_idx, casci.l2_idx, casci.l2_idx)] = 2 * casci.exchange

    c2 = .5 * (c2 + c2.transpose(2, 3, 0, 1))

    c_eff = np.einsum('pqrs,pqrs', casci.exchange,
                      np.einsum('pq,rs->pqrs', casci.one_rdm_l1, casci.one_rdm_l2))
    return c_eff, c1, c2


def get_hd_ints(casci):
    c1 = np.zeros_like(casci.c1)
    c2 = np.zeros_like(casci.c2)

    c1[np.ix_(*[casci.l1_idx]*2)] = casci.c1[np.ix_(*[casci.l1_idx]*2)]
    c1[np.ix_(*[casci.l2_idx]*2)] = casci.c1[np.ix_(*[casci.l2_idx]*2)]

    c2[np.ix_(*[casci.l1_idx]*4)] = casci.c2[np.ix_(*[casci.l1_idx]*4)]
    c2[np.ix_(*[casci.l2_idx]*4)] = casci.c2[np.ix_(*[casci.l2_idx]*4)]

    c2[np.ix_(casci.l1_idx, casci.l1_idx, casci.l2_idx, casci.l2_idx)] = 2 * casci.exchange

    c2 = .5 * (c2 + c2.transpose(2, 3, 0, 1))
    return casci.c0 + casci.nuclear_repulsion, c1, c2


def get_h1ct_ints(casci):
    c1 = np.zeros_like(casci.c1)
    c2 = np.zeros_like(casci.c2)

    c1[np.ix_(casci.l1_idx, casci.l2_idx)] = casci.c1[np.ix_(casci.l1_idx, casci.l2_idx)]
    # - np.einsum('prrq->pq',c2[np.ix_(casci.l1_idx, casci.l1_idx, casci.l1_idx, casci.l2_idx)])
    c1[np.ix_(casci.l2_idx, casci.l1_idx)] = casci.c1[np.ix_(casci.l2_idx, casci.l1_idx)]
    # - np.einsum('prrq->pq',c2[np.ix_(casci.l2_idx, casci.l2_idx, casci.l2_idx, casci.l1_idx)])

    c2[np.ix_(casci.l1_idx, casci.l1_idx, casci.l1_idx, casci.l2_idx)] = 2 * casci.c2[
        np.ix_(casci.l1_idx, casci.l1_idx, casci.l1_idx, casci.l2_idx)]
    c2[np.ix_(casci.l1_idx, casci.l1_idx, casci.l2_idx, casci.l1_idx)] = 2 * casci.c2[
        np.ix_(casci.l1_idx, casci.l1_idx, casci.l1_idx, casci.l2_idx)].transpose(0, 1, 3, 2)

    c2[np.ix_(casci.l2_idx, casci.l2_idx, casci.l2_idx, casci.l1_idx)] = 2 * casci.c2[
        np.ix_(casci.l2_idx, casci.l2_idx, casci.l2_idx, casci.l1_idx)]
    c2[np.ix_(casci.l2_idx, casci.l2_idx, casci.l1_idx, casci.l2_idx)] = 2 * casci.c2[
        np.ix_(casci.l2_idx, casci.l2_idx, casci.l2_idx, casci.l1_idx)].transpose(0, 1, 3, 2)

    c2 = .5 * (c2 + c2.transpose(2, 3, 0, 1))
    return 0., c1, c2


def get_h2ct_ints(casci):
    c1 = np.zeros_like(casci.c1)
    c2 = np.zeros_like(casci.c2)

    c2[np.ix_(casci.l1_idx, casci.l2_idx, casci.l1_idx, casci.l2_idx)] = casci.c2[
        np.ix_(casci.l1_idx, casci.l2_idx, casci.l1_idx, casci.l2_idx)]
    c2[np.ix_(casci.l2_idx, casci.l1_idx, casci.l2_idx, casci.l1_idx)] = casci.c2[
        np.ix_(casci.l2_idx, casci.l1_idx, casci.l2_idx, casci.l1_idx)]
    return 0., c1, c2


def get_htt_ints(casci):
    c1 = np.zeros_like(casci.c1)
    c2a = np.zeros_like(casci.c2)
    c2b = np.zeros_like(casci.c2)

    c2a[np.ix_(casci.l1_idx, casci.l2_idx, casci.l2_idx, casci.l1_idx)] = 2 * casci.c2[
        np.ix_(casci.l1_idx, casci.l2_idx, casci.l2_idx, casci.l1_idx)]

    c2b[np.ix_(casci.l1_idx, casci.l1_idx, casci.l2_idx, casci.l2_idx)] = casci.c2[
        np.ix_(casci.l1_idx, casci.l2_idx, casci.l2_idx, casci.l1_idx)].transpose(0, 3, 2, 1)

    c2a = .5 * (c2a + c2a.transpose(2, 3, 0, 1))
    c2b = .5 * (c2b + c2b.transpose(2, 3, 0, 1))

    return 0., c1, c2a + c2b


def get_ptinf_contributions(casci):
    ncas, nelecas = casci.ncas, casci.nelecas

    c0_0, c1_0, c2_0 = get_h0_ints(casci)
    c0_disp, c1_disp, c2_disp = get_hdisp_ints(casci)
    c0_1ct, c1_1ct, c2_1ct = get_h1ct_ints(casci)
    c0_2ct, c1_2ct, c2_2ct = get_h2ct_ints(casci)
    c0_tt, c1_tt, c2_tt = get_htt_ints(casci)

    e_ci_0, civec_0 = run_fci(c1_0, c2_0, ncas, nelecas)
    e_ci_disp, civec_disp = run_fci(c1_0 + c1_disp, c2_0 + c2_disp, ncas, nelecas)
    e_ci_1ct, civec_1ct = run_fci(c1_0 + c1_1ct, c2_0 + c2_1ct, ncas, nelecas)
    e_ci_2ct, civec_2ct = run_fci(c1_0 + c1_2ct, c2_0 + c2_2ct, ncas, nelecas)
    e_ci_tt, civec_tt = run_fci(c1_0 + c1_tt, c2_0 + c2_tt, ncas, nelecas)

    c1_tot = c1_0 + c1_disp + c1_1ct + c1_2ct + c1_tt
    c2_tot = c2_0 + c2_disp + c2_1ct + c2_2ct + c2_tt

    assert np.allclose(c1_tot, casci.c1)
    assert np.allclose(c2_tot, casci.c2)

    energies = np.array([e_ci_0, e_ci_disp, e_ci_1ct, e_ci_2ct, e_ci_tt])
    civecs = (civec_0, civec_disp, civec_1ct, civec_2ct, civec_tt)

    return c0_0, c0_disp, energies, civecs
