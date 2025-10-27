#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Created on Sat Mar  9 16:45:23 2024

@author: emielkoridon
Functions that need to be defined in every module in ../perturbations:
    - get_h0
    - get_ovlp
    - get_h_prime


Computes second-order perturbation energy for dispersion interactions.
Shares k-RDMs up to k=4.

Uses the perturbing functions:

.. math::
    \{ |\Psi_{tuvw} \rangle = E_{t_A u_A} E_{v_B w_B} |\Psi^0\rangle \}

"""

import numpy as np

# MAIN FUNCTIONS


def get_h0(indices, rdms, integrals):
    pass


def get_ovlp(indices, rdms):
    pass


def get_h_prime(indices, rdms, integrals):
    pass


def get_ct_frag_rdms(ct):
    if ct == 'A':
        zero_ct_rdm_l1 = zero_ct2_rdm('l1')
        one_ct_rdm_l1 = one_ct2_rdm('l1')
        two_ct_rdm_l1 = two_ct2_rdm('l1')
        zero_ct_rdm_l2 = .5 * one_rdm_l2
        one_ct_rdm_l2 = one_ct1_rdm('l2')
        two_ct_rdm_l2 = two_ct1_rdm('l2')
    elif ct == 'B':
        zero_ct_rdm_l1 = .5 * one_rdm_l1
        one_ct_rdm_l1 = one_ct1_rdm('l1')
        two_ct_rdm_l1 = two_ct1_rdm('l1')
        zero_ct_rdm_l2 = zero_ct2_rdm('l2')
        one_ct_rdm_l2 = one_ct2_rdm('l2')
        two_ct_rdm_l2 = two_ct2_rdm('l2')

    return (zero_ct_rdm_l1, one_ct_rdm_l1, two_ct_rdm_l1,
            zero_ct_rdm_l2, one_ct_rdm_l2, two_ct_rdm_l2)


def get_h0_pure_ct(fragment, ct):
    (zero_ct_rdm_l1, one_ct_rdm_l1, two_ct_rdm_l1,
     zero_ct_rdm_l2, one_ct_rdm_l2, two_ct_rdm_l2) = get_ct_frag_rdms(ct)

    if fragment == 'l1':
        c1_pure = c1[np.ix_(*[l1_idx]*2)]
        c2_pure = c2[np.ix_(*[l1_idx]*4)]
        c1 = h_prime(c1_pure, c2_pure)

        if ct == 'A':
            h1 = 2 * np.einsum('pq, tkpq, lu->kltu', c1, one_ct_rdm_l1, zero_ct_rdm_l2)
            h2 = np.einsum('pqrs, tkpqrs, lu->kltu',
                           c2_pure, two_ct_rdm_l1, zero_ct_rdm_l2)
        elif ct == 'B':
            h1 = 2 * np.einsum('pq, lupq, tk->kltu', c1, one_ct_rdm_l1, zero_ct_rdm_l2)
            h2 = np.einsum('pqrs, lupqrs, tk->kltu',
                           c2_pure, two_ct_rdm_l1, zero_ct_rdm_l2)
    elif fragment == 'l2':
        c1_pure = c1[np.ix_(*[l2_idx]*2)]
        c2_pure = c2[np.ix_(*[l2_idx]*4)]
        c1 = h_prime(c1_pure, c2_pure)

        if ct == 'A':
            h1 = 2 * np.einsum('pq, lupq, tk->kltu', c1, one_ct_rdm_l2, zero_ct_rdm_l1)
            h2 = np.einsum('pqrs, lupqrs, tk->kltu',
                           c2_pure, two_ct_rdm_l2, zero_ct_rdm_l1)
        elif ct == 'B':
            h1 = 2 * np.einsum('pq, tkpq, lu->kltu', c1, one_ct_rdm_l2, zero_ct_rdm_l1)
            h2 = np.einsum('pqrs, tkpqrs, lu->kltu',
                           c2_pure, two_ct_rdm_l2, zero_ct_rdm_l1)
    return h1 + h2


def get_h0_disp_ct(ct):
    (zero_ct_rdm_l1, one_ct_rdm_l1, two_ct_rdm_l1,
     zero_ct_rdm_l2, one_ct_rdm_l2, two_ct_rdm_l2) = get_ct_frag_rdms(ct)

    if ct == 'A':
        h1 = 2 * np.einsum('pqrs, rs, tkpq, lu->kltu', exchange,
                           one_rdm_l2, one_ct_rdm_l1, zero_ct_rdm_l2)
        h2 = 2 * np.einsum('pqrs, pq, tk, lurs->kltu', exchange,
                           one_rdm_l1, zero_ct_rdm_l1, one_ct_rdm_l2)
    if ct == 'B':
        h1 = 2 * np.einsum('pqrs, rs, lupq, tk->kltu', exchange,
                           one_rdm_l2, one_ct_rdm_l1, zero_ct_rdm_l2)
        h2 = 2 * np.einsum('pqrs, pq, lu, tkrs->kltu', exchange,
                           one_rdm_l1, zero_ct_rdm_l1, one_ct_rdm_l2)

    h3 = - np.einsum('pqrs, pq, rs', exchange,
                     one_rdm_l1, one_rdm_l2) * get_overlap_ct(ct)
    return h1 + h2 + h3


def get_overlap_ct(ct):
    if ct == 'A':
        return np.einsum('kt, lu->kltu', zero_ct2_rdm('l1'), one_rdm_l2)
    elif ct == 'B':
        return np.einsum('lu, kt->kltu', one_rdm_l1, zero_ct2_rdm('l2'))


def get_ham_ct_col(ct):
    (zero_ct_rdm_l1, _, _,
     zero_ct_rdm_l2, _, _) = get_ct_frag_rdms(ct)
    g_eff_A = c2[np.ix_(l1_idx, l1_idx, l1_idx, l2_idx)]
    g_eff_B = c2[np.ix_(l2_idx, l2_idx, l2_idx, l1_idx)]
    if ct == 'A':
        one_rdm_ct_l1 = - .5 * two_rdm_l1 + np.einsum(
            'pq, tr->pqtr', one_rdm_l1, np.eye(ncas_l1))
        h_eff = c1[np.ix_(l2_idx, l1_idx)] - np.einsum(
            'prrq->pq', c2[np.ix_(l2_idx, l2_idx, l2_idx, l1_idx)])

        h1 = np.einsum('pq, tq, pu->tu', h_eff, zero_ct_rdm_l1, one_rdm_l2)
        h2_A = np.einsum('pqrs, pqtr, su->tu', g_eff_A,
                         one_rdm_ct_l1, one_rdm_l2)
        h2_B = np.einsum('pqrs, ts, pqru->tu', g_eff_B,
                         zero_ct_rdm_l1, two_rdm_l2)
    elif ct == 'B':
        one_rdm_ct_l2 = - .5 * two_rdm_l2 + np.einsum(
            'pq, tr->pqtr', one_rdm_l2, np.eye(ncas_l2))
        h_eff = c1[np.ix_(l1_idx, l2_idx)] - np.einsum(
            'prrq->pq', c2[np.ix_(l1_idx, l1_idx, l1_idx, l2_idx)])

        h1 = np.einsum('pq, pu, tq->tu', h_eff, one_rdm_l1, zero_ct_rdm_l2)
        h2_A = np.einsum('pqrs, pqru, ts->tu', g_eff_A,
                         two_rdm_l1, zero_ct_rdm_l2)
        h2_B = np.einsum('pqrs, su, pqtr->tu', g_eff_B,
                         one_rdm_l1, one_rdm_ct_l2)
    return h1 + h2_A + h2_B


def run_fragpt2_ct():
    get_four_rdms()

    nex_A = ncas_l1 * ncas_l2
    E_0 = h0_expval(w_core=False)
    S_ovlp_A = get_overlap_ct('A').reshape((nex_A, nex_A))
    H_l1_A = get_h0_pure_ct('l1', 'A')
    H_l2_A = get_h0_pure_ct('l2', 'A')
    H_0_ct_A = get_h0_disp_ct('A')
    H_0_A = H_l1_A + H_l2_A + H_0_ct_A
    H_0_A = H_0_A.reshape((nex_A, nex_A))

    print("Hermitian check A, should be true:")
    print(np.allclose(H_l1_A, np.einsum('kltu->tukl', H_l1_A)))
    print(np.allclose(H_l2_A, np.einsum('kltu->tukl', H_l2_A)))
    print(np.allclose(H_0_ct_A, np.einsum('kltu->tukl', H_0_ct_A)))
    print(np.allclose(H_0_A, H_0_A.T))

    H_prime_col_A = get_ham_ct_col('A')
    H_prime_col_A = H_prime_col_A.reshape((nex_A))

    psi1_c_ct_A = np.linalg.solve((H_0_A - E_0 * S_ovlp_A),
                                  -H_prime_col_A)

    e_pt2_A = np.einsum('n, n', H_prime_col_A,
                        psi1_c_ct_A)

    nex_B = ncas_l2 * ncas_l1
    S_ovlp_B = get_overlap_ct('B').reshape((nex_B, nex_B))
    H_l1_B = get_h0_pure_ct('l1', 'B')
    H_l2_B = get_h0_pure_ct('l2', 'B')
    H_0_ct_B = get_h0_disp_ct('B')
    H_0_B = H_l1_B + H_l2_B + H_0_ct_B
    H_0_B = H_0_B.reshape((nex_B, nex_B))

    H_prime_col_B = get_ham_ct_col('B')
    H_prime_col_B = H_prime_col_B.reshape((nex_B))

    print("Hermitian check B, should be true:")
    print(np.allclose(H_l1_B, np.einsum('kltu->tukl', H_l1_B)))
    print(np.allclose(H_l2_B, np.einsum('kltu->tukl', H_l2_B)))
    print(np.allclose(H_0_ct_B, np.einsum('kltu->tukl', H_0_ct_B)))
    print(np.allclose(H_0_B, H_0_B.T))

    psi1_c_ct_B = np.linalg.solve((H_0_B - E_0 * S_ovlp_B),
                                  -H_prime_col_B)

    e_pt2_B = np.einsum('n, n', H_prime_col_B,
                        psi1_c_ct_B)

    return e_pt2_A, e_pt2_B
