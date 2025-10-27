#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Created on Tue Apr 16 12:24:07 2024

@author: emielkoridon

Functions that need to be defined in every module in ../perturbations:
    - get_h0
    - get_ovlp
    - get_h_prime


Computes second-order perturbation energy for dispersion interactions.
Needs k-RDMs up to k=4.

Uses the perturbing functions:

.. math::
    \{ |\Psi_{tuvw} \rangle = E_{t_A u_A} E_{v_B w_B} |\Psi^0\rangle \}

"""

import numpy as np
from fragpt2.fragci import get_exchange


# HELPER FUNCTIONS

def get_ncas(indices):
    l1_idx, l2_idx = indices
    ncas_l1 = len(l1_idx)
    ncas_l2 = len(l2_idx)
    return ncas_l1, ncas_l2


def get_nex(indices):
    ncas_l1, ncas_l2 = get_ncas(indices)
    return ncas_l1 * ncas_l1 * ncas_l2 * ncas_l2


# MAIN FUNCTIONS

def get_h0(indices, rdms, integrals):
    c1, c2 = integrals
    exchange = get_exchange(indices, integrals)
    (one_rdm_l1, two_rdm_l1, three_rdm_l1, four_rdm_l1,
     one_rdm_l2, two_rdm_l2, three_rdm_l2, four_rdm_l2) = rdms
    nex = get_nex(indices)

    H_l1 = get_pure_ham('l1', indices, rdms, integrals)
    H_l2 = get_pure_ham('l2', indices, rdms, integrals)
    H_0_disp = get_h0_disp(rdms, exchange)
    H_0 = H_l1 + H_l2 + H_0_disp
    H_0 = H_0.reshape((nex, nex))
    return H_0


def get_ovlp(indices, rdms):
    _, two_rdm_l1, _, _, _, two_rdm_l2, _, _ = rdms
    nex = get_nex(indices)
    S_ovlp = np.einsum('lktu,nmvw->klmntuvw',
                       two_rdm_l1, two_rdm_l2)
    S_ovlp = S_ovlp.reshape((nex, nex))
    return S_ovlp


def get_h_prime(indices, rdms, integrals):
    exchange = get_exchange(indices, integrals)
    (one_rdm_l1, two_rdm_l1, three_rdm_l1, four_rdm_l1,
     one_rdm_l2, two_rdm_l2, three_rdm_l2, four_rdm_l2) = rdms
    nex = get_nex(indices)
    H_0_disp_col = get_h0_disp_col(
        one_rdm_l1, one_rdm_l2, two_rdm_l1, two_rdm_l2, exchange)
    H_prime_col = get_ham_disp_col(
        two_rdm_l1, two_rdm_l2, exchange) - H_0_disp_col
    H_prime_col = H_prime_col.reshape((nex))
    return H_prime_col


# UTILITY FUNCTIONS

def get_pure_ham(fragment, indices, rdms, integrals):
    l1_idx, l2_idx = indices
    c1, c2 = integrals
    (one_rdm_l1, two_rdm_l1, three_rdm_l1, four_rdm_l1,
     one_rdm_l2, two_rdm_l2, three_rdm_l2, four_rdm_l2) = rdms
    if fragment == 'l1':
        c1_pure = c1[np.ix_(*[l1_idx]*2)]
        c2_pure = c2[np.ix_(*[l1_idx]*4)]

        h1 = np.einsum('pq, lkpqtu->kltu', c1_pure, three_rdm_l1)
        h2_c = .5 * np.einsum('pqrs, lkpqrstu->kltu', c2_pure, four_rdm_l1)
        h2_ex = -.5 * np.einsum('prrq, lkpqtu->kltu', c2_pure, three_rdm_l1)

        return np.einsum('kltu, nmvw->klmntuvw', h1 + h2_c + h2_ex, two_rdm_l2)

    elif fragment == 'l2':
        c1_pure = c1[np.ix_(*[l2_idx]*2)]
        c2_pure = c2[np.ix_(*[l2_idx]*4)]

        h1 = np.einsum('pq, nmpqvw->mnvw', c1_pure, three_rdm_l2)
        h2_c = .5 * np.einsum('pqrs, nmpqrsvw->mnvw', c2_pure, four_rdm_l2)
        h2_ex = -.5 * np.einsum('prrq, nmpqvw->mnvw', c2_pure, three_rdm_l2)

        return np.einsum('lktu, mnvw->klmntuvw', two_rdm_l1, h1 + h2_c + h2_ex)
    else:
        raise ValueError(f'fragment {fragment} doesnt exist')


def get_h0_disp(rdms, exchange):
    (one_rdm_l1, two_rdm_l1, three_rdm_l1, _,
     one_rdm_l2, two_rdm_l2, three_rdm_l2, _) = rdms

    h1 = np.einsum('pqrs, rs, lkpqtu, nmvw->klmntuvw', exchange,
                   one_rdm_l2, three_rdm_l1, two_rdm_l2)
    h2 = np.einsum('pqrs, pq, lktu, nmrsvw->klmntuvw', exchange,
                   one_rdm_l1, two_rdm_l1, three_rdm_l2)
    h3 = - np.einsum('pqrs, pq, rs, lktu, nmvw->klmntuvw', exchange,
                     one_rdm_l1, one_rdm_l2, two_rdm_l1, two_rdm_l2)
    return h1 + h2 + h3


def get_h0_disp_col(one_rdm_l1, one_rdm_l2, two_rdm_l1, two_rdm_l2, exchange):
    h1 = np.einsum('rs, pqtu, vw->pqrstuvw',
                   one_rdm_l2, two_rdm_l1, one_rdm_l2)
    h2 = np.einsum('pq, tu, rsvw->pqrstuvw',
                   one_rdm_l1, one_rdm_l1, two_rdm_l2)
    h3 = - np.einsum('pq, rs, tu, vw->pqrstuvw',
                     one_rdm_l1, one_rdm_l2,
                     one_rdm_l1, one_rdm_l2)
    return np.einsum('pqrs, pqrstuvw->tuvw', exchange, h1 + h2 + h3)


def get_ham_disp_col(two_rdm_l1, two_rdm_l2, exchange):
    return np.einsum('pqrs, pqtu, rsvw->tuvw',
                     exchange, two_rdm_l1, two_rdm_l2)


# Some stuff that could be nice for GEV problem and PT3:


# def get_ham_disp(exchange, three_rdm_l1, three_rdm_l2):
#     return np.einsum('pqrs, lkpqtu, nmrsvw->klmntuvw',
#                      exchange, three_rdm_l1, three_rdm_l2)


# def run():
#     S_ovlp = get_disp_overlap(w_zero=False)

#     H_l1 = get_pure_ham('l1')  # .reshape((nex, nex))
#     H_l2 = get_pure_ham('l2')  # .reshape((nex, nex))
#     H_0_disp = get_h0_disp()  # .reshape((nex, nex))
#     H_0 = H_l1 + H_l2 + H_0_disp
#     H_0 = H_0.reshape((nex, nex))

#     H_0_disp_col = get_h0_disp_col()

#     H_prime_col = get_ham_disp_col() - H_0_disp_col

#     H_prime_col = H_prime_col.reshape((nex))

#     # H_prime_eig = eigvec.T @ H_prime_col

#     psi1_c = np.linalg.solve((H_0 - E_0 * S_ovlp),
#                              -H_prime_col)

#     e_pt2 = np.einsum('n, n', H_prime_col,
#                       psi1_c)
#     return e_pt2


# def run_pt3():
#     e_pt2 = run_fragpt2()
#     E_0 = h0_expval(w_core=False)
#     S_ovlp = get_wfn_overlap(w_zero=False)
#     S_ovlp_col = np.einsum('tu, vw->tuvw',
#                            one_rdm_l1, one_rdm_l2).reshape((nex))

#     H_l1 = get_pure_ham('l1')  # .reshape((nex, nex))
#     H_l2 = get_pure_ham('l2')  # .reshape((nex, nex))
#     H_0_disp = get_h0_disp()  # .reshape((nex, nex))
#     H_0 = H_l1 + H_l2 + H_0_disp
#     H_0 = H_0.reshape((nex, nex))

#     H_0_disp_col = get_h0_disp_col()

#     eigval, eigvec = scipy.linalg.eigh(H_0, b=S_ovlp)

#     H_prime = get_ham_disp() - H_0_disp
#     H_prime = H_prime.reshape((nex, nex))

#     H_prime_col = (get_ham_disp_col() - H_0_disp_col)
#     H_prime_col = H_prime_col.reshape((nex))

#     H_prime_eig = eigvec.T @ H_prime @ eigvec
#     H_prime_col_eig = eigvec.T @ H_prime_col

#     S_ovlp_col_eig = eigvec.T @ S_ovlp_col

#     assert np.allclose(E_0, eigval[0], atol=1e-6)

#     norm = E_0 - eigval
#     norm[0] = 1e99

#     H_prime_col_norm = np.divide(H_prime_col_eig, norm)

#     e_pt3 = sum((
#         np.einsum('n, nm, m', H_prime_col_norm, H_prime_eig, H_prime_col_norm),
#         - e_pt2 * (H_prime_col_norm @ S_ovlp_col_eig)))
#     return e_pt3, e_pt2


# def solve_gev():
#     get_four_rdms()
#     S_ovlp = get_wfn_overlap(w_zero=False)
#     H_l1 = get_pure_ham('l1')
#     H_l2 = get_pure_ham('l2')
#     H_disp = get_ham_disp()
#     H_full = H_l1 + H_l2 + H_disp
#     H_full = H_full.reshape((nex, nex))

#     v, w = scipy.linalg.eigh(H_full, b=S_ovlp, lower=True)

#     return v[0] + c0 + nuclear_repulsion
