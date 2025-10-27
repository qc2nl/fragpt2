#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Created on Tue Apr 16 12:24:29 2024

@author: emielkoridon
Functions that need to be defined in every module in ../perturbations:
    - get_h0
    - get_ovlp
    - get_h_prime


Computes second-order perturbation energy for dispersion interactions.
Needs k-RDMs up to k=5.

Uses the perturbing functions:

.. math::
    |\Psi^0_{tuvw}\rangle \equiv E_{tu}E_{vw} |\Psi^0\rangle
    \begin{cases}
     & tuv \in A, w \in B\\
     & tuw \in A, v \in B\\
     & tuv \in B, w \in A\\
     & tuw \in B, v \in A
    \end{cases}

"""

import numpy as np
from fragpt2.fragci import get_exchange

# HELPER FUNCTIONS


def h_prime(one_body_integrals, two_body_integrals):
    return one_body_integrals - .5 * np.einsum('prrq->pq', two_body_integrals)


def get_ncas(indices):
    l1_idx, l2_idx = indices
    ncas_l1 = len(l1_idx)
    ncas_l2 = len(l2_idx)
    return ncas_l1, ncas_l2


def get_nex(indices):
    ncas_l1, ncas_l2 = get_ncas(indices)
    return ncas_l1 * ncas_l1 * ncas_l1 * ncas_l2, ncas_l2 * ncas_l2 * ncas_l1 * ncas_l2


# MAIN FUNCTIONS

def get_h0(ct, indices, rdms, ct_rdms, integrals):
    nex_AA, nex_BB = get_nex(indices)

    H_l1_AA_AA = get_h0_pure_ct('l1', ct, 'A', 'A', indices, ct_rdms,
                                integrals).reshape((nex_AA, nex_AA))
    H_l1_AA_BB = get_h0_pure_ct('l1', ct, 'A', 'B', indices, ct_rdms,
                                integrals).reshape((nex_AA, nex_BB))
    H_l1_BB_AA = get_h0_pure_ct('l1', ct, 'B', 'A', indices, ct_rdms,
                                integrals).reshape((nex_BB, nex_AA))
    H_l1_BB_BB = get_h0_pure_ct('l1', ct, 'B', 'B', indices, ct_rdms,
                                integrals).reshape((nex_BB, nex_BB))

    H_l2_AA_AA = get_h0_pure_ct('l2', ct, 'A', 'A', indices, ct_rdms,
                                integrals).reshape((nex_AA, nex_AA))
    H_l2_AA_BB = get_h0_pure_ct('l2', ct, 'A', 'B', indices, ct_rdms,
                                integrals).reshape((nex_AA, nex_BB))
    H_l2_BB_AA = get_h0_pure_ct('l2', ct, 'B', 'A', indices, ct_rdms,
                                integrals).reshape((nex_BB, nex_AA))
    H_l2_BB_BB = get_h0_pure_ct('l2', ct, 'B', 'B', indices, ct_rdms,
                                integrals).reshape((nex_BB, nex_BB))

    H_0_ct_AA_AA = get_h0_disp_ct(ct, 'A', 'A', indices, rdms, ct_rdms,
                                  integrals).reshape((nex_AA, nex_AA))
    H_0_ct_AA_BB = get_h0_disp_ct(ct, 'A', 'B', indices, rdms, ct_rdms,
                                  integrals).reshape((nex_AA, nex_BB))
    H_0_ct_BB_AA = get_h0_disp_ct(ct, 'B', 'A', indices, rdms, ct_rdms,
                                  integrals).reshape((nex_BB, nex_AA))
    H_0_ct_BB_BB = get_h0_disp_ct(ct, 'B', 'B', indices, rdms, ct_rdms,
                                  integrals).reshape((nex_BB, nex_BB))

    H_l1 = np.block([[H_l1_AA_AA, H_l1_AA_BB],
                     [H_l1_BB_AA, H_l1_BB_BB]])
    H_l2 = np.block([[H_l2_AA_AA, H_l2_AA_BB],
                     [H_l2_BB_AA, H_l2_BB_BB]])
    H_0_ct = np.block([[H_0_ct_AA_AA, H_0_ct_AA_BB],
                       [H_0_ct_BB_AA, H_0_ct_BB_BB]])

    H_0 = H_l1 + H_l2 + H_0_ct

    return H_0


def get_ovlp(ct, indices, ct_rdms):
    nex_AA, nex_BB = get_nex(indices)

    S_ovlp_AA_AA = get_overlap_ct(
        ct, 'A', 'A', ct_rdms).reshape((nex_AA, nex_AA))
    S_ovlp_AA_BB = get_overlap_ct(
        ct, 'A', 'B', ct_rdms).reshape((nex_AA, nex_BB))
    S_ovlp_BB_AA = get_overlap_ct(
        ct, 'B', 'A', ct_rdms).reshape((nex_BB, nex_AA))
    S_ovlp_BB_BB = get_overlap_ct(
        ct, 'B', 'B', ct_rdms).reshape((nex_BB, nex_BB))
    S_ovlp = np.block([[S_ovlp_AA_AA, S_ovlp_AA_BB],
                       [S_ovlp_BB_AA, S_ovlp_BB_BB]])
    return S_ovlp


def get_h_prime(ct, indices, ct_rdms, integrals):
    nex_AA, nex_BB = get_nex(indices)

    H_prime_col_AA = get_ham_ct_col(ct, 'A', indices, ct_rdms, integrals)
    H_prime_col_AA = H_prime_col_AA.reshape((nex_AA))
    H_prime_col_BB = get_ham_ct_col(ct, 'B', indices, ct_rdms, integrals)
    H_prime_col_BB = H_prime_col_BB.reshape((nex_BB))
    H_prime_col = np.concatenate((H_prime_col_AA, H_prime_col_BB))

    return H_prime_col


# UTILITY FUNCTIONS

def get_h0_pure_ct(fragment, ct, localex_l, localex_r, indices, ct_rdms, integrals):
    l1_idx, l2_idx = indices
    (zero_ct_rdm_l1, one_ct_rdm_l1, two_ct_rdm_l1, three_ct_rdm_l1, four_ct_rdm_l1,
     zero_ct_rdm_l2, one_ct_rdm_l2, two_ct_rdm_l2, three_ct_rdm_l2,
     four_ct_rdm_l2) = ct_rdms[ct]
    c1, c2 = integrals

    if fragment == 'l1':
        c1_pure = c1[np.ix_(*[l1_idx]*2)]
        c2_pure = c2[np.ix_(*[l1_idx]*4)]
        c1_prime = h_prime(c1_pure, c2_pure)

        if ct == 'A':
            if localex_l == 'A' and localex_r == 'A':
                h1 = 2 * np.einsum('pq, vmlkpqtu, nw->klmntuvw', c1_prime, three_ct_rdm_l1,
                                   zero_ct_rdm_l2)
                h2 = np.einsum('pqrs, vmlkpqrstu, nw->klmntuvw',
                               c2_pure, four_ct_rdm_l1, zero_ct_rdm_l2)
            elif localex_l == 'A' and localex_r == 'B':
                h1 = 2 * np.einsum('pq, vmlkpq, nwtu->klmntuvw', c1_prime, two_ct_rdm_l1,
                                   one_ct_rdm_l2)
                h2 = np.einsum('pqrs, vmlkpqrs, nwtu->klmntuvw',
                               c2_pure, three_ct_rdm_l1, one_ct_rdm_l2)
            elif localex_l == 'B' and localex_r == 'A':
                h1 = 2 * np.einsum('pq, vmpqtu, nwlk->klmntuvw', c1_prime, two_ct_rdm_l1,
                                   one_ct_rdm_l2)
                h2 = np.einsum('pqrs, vmpqrstu, nwlk->klmntuvw',
                               c2_pure, three_ct_rdm_l1, one_ct_rdm_l2)
            elif localex_l == 'B' and localex_r == 'B':
                h1 = 2 * np.einsum('pq, vmpq, nwlktu->klmntuvw', c1_prime, one_ct_rdm_l1,
                                   two_ct_rdm_l2)
                h2 = np.einsum('pqrs, vmpqrs, nwlktu->klmntuvw',
                               c2_pure, two_ct_rdm_l1, two_ct_rdm_l2)
        elif ct == 'B':
            if localex_l == 'A' and localex_r == 'A':
                h1 = 2 * np.einsum('pq, nwlkpqtu, vm->klmntuvw', c1_prime, three_ct_rdm_l1,
                                   zero_ct_rdm_l2)
                h2 = np.einsum('pqrs, nwlkpqrstu, vm->klmntuvw',
                               c2_pure, four_ct_rdm_l1, zero_ct_rdm_l2)
            elif localex_l == 'A' and localex_r == 'B':
                h1 = 2 * np.einsum('pq, nwlkpq, vmtu->klmntuvw', c1_prime, two_ct_rdm_l1,
                                   one_ct_rdm_l2)
                h2 = np.einsum('pqrs, nwlkpqrs, vmtu->klmntuvw',
                               c2_pure, three_ct_rdm_l1, one_ct_rdm_l2)
            elif localex_l == 'B' and localex_r == 'A':
                h1 = 2 * np.einsum('pq, nwpqtu, vmlk->klmntuvw', c1_prime, two_ct_rdm_l1,
                                   one_ct_rdm_l2)
                h2 = np.einsum('pqrs, nwpqrstu, vmlk->klmntuvw',
                               c2_pure, three_ct_rdm_l1, one_ct_rdm_l2)
            elif localex_l == 'B' and localex_r == 'B':
                h1 = 2 * np.einsum('pq, nwpq, vmlktu->klmntuvw', c1_prime, one_ct_rdm_l1,
                                   two_ct_rdm_l2)
                h2 = np.einsum('pqrs, nwpqrs, vmlktu->klmntuvw',
                               c2_pure, two_ct_rdm_l1, two_ct_rdm_l2)
    elif fragment == 'l2':
        c1_pure = c1[np.ix_(*[l2_idx]*2)]
        c2_pure = c2[np.ix_(*[l2_idx]*4)]
        c1_prime = h_prime(c1_pure, c2_pure)

        if ct == 'A':
            if localex_l == 'A' and localex_r == 'A':
                h1 = 2 * np.einsum('pq, vmlktu, nwpq->klmntuvw', c1_prime, two_ct_rdm_l1,
                                   one_ct_rdm_l2)
                h2 = np.einsum('pqrs, vmlktu, nwpqrs->klmntuvw',
                               c2_pure, two_ct_rdm_l1, two_ct_rdm_l2)
            elif localex_l == 'A' and localex_r == 'B':
                h1 = 2 * np.einsum('pq, vmlk, nwpqtu->klmntuvw', c1_prime, one_ct_rdm_l1,
                                   two_ct_rdm_l2)
                h2 = np.einsum('pqrs, vmlk, nwpqrstu->klmntuvw',
                               c2_pure, one_ct_rdm_l1, three_ct_rdm_l2)
            elif localex_l == 'B' and localex_r == 'A':
                h1 = 2 * np.einsum('pq, vmtu, nwlkpq->klmntuvw', c1_prime, one_ct_rdm_l1,
                                   two_ct_rdm_l2)
                h2 = np.einsum('pqrs, vmtu, nwlkpqrs->klmntuvw',
                               c2_pure, one_ct_rdm_l1, three_ct_rdm_l2)
            elif localex_l == 'B' and localex_r == 'B':
                h1 = 2 * np.einsum('pq, vm, nwlkpqtu->klmntuvw', c1_prime, zero_ct_rdm_l1,
                                   three_ct_rdm_l2)
                h2 = np.einsum('pqrs, vm, nwlkpqrstu->klmntuvw',
                               c2_pure, zero_ct_rdm_l1, four_ct_rdm_l2)
        elif ct == 'B':
            if localex_l == 'A' and localex_r == 'A':
                h1 = 2 * np.einsum('pq, nwlktu, vmpq->klmntuvw', c1_prime, two_ct_rdm_l1,
                                   one_ct_rdm_l2)
                h2 = np.einsum('pqrs, nwlktu, vmpqrs->klmntuvw',
                               c2_pure, two_ct_rdm_l1, two_ct_rdm_l2)
            elif localex_l == 'A' and localex_r == 'B':
                h1 = 2 * np.einsum('pq, nwlk, vmpqtu->klmntuvw', c1_prime, one_ct_rdm_l1,
                                   two_ct_rdm_l2)
                h2 = np.einsum('pqrs, nwlk, vmpqrstu->klmntuvw',
                               c2_pure, one_ct_rdm_l1, three_ct_rdm_l2)
            elif localex_l == 'B' and localex_r == 'A':
                h1 = 2 * np.einsum('pq, nwtu, vmlkpq->klmntuvw', c1_prime, one_ct_rdm_l1,
                                   two_ct_rdm_l2)
                h2 = np.einsum('pqrs, nwtu, vmlkpqrs->klmntuvw',
                               c2_pure, one_ct_rdm_l1, three_ct_rdm_l2)
            elif localex_l == 'B' and localex_r == 'B':
                h1 = 2 * np.einsum('pq, nw, vmlkpqtu->klmntuvw', c1_prime, zero_ct_rdm_l1,
                                   three_ct_rdm_l2)
                h2 = np.einsum('pqrs, nw, vmlkpqrstu->klmntuvw',
                               c2_pure, zero_ct_rdm_l1, four_ct_rdm_l2)
    return h1 + h2


def get_h0_disp_ct(ct, localex_l, localex_r, indices, rdms, ct_rdms, integrals):
    l1_idx, l2_idx = indices
    (zero_ct_rdm_l1, one_ct_rdm_l1, two_ct_rdm_l1, three_ct_rdm_l1, four_ct_rdm_l1,
     zero_ct_rdm_l2, one_ct_rdm_l2, two_ct_rdm_l2, three_ct_rdm_l2,
     four_ct_rdm_l2) = ct_rdms[ct]
    c1, c2 = integrals
    exchange = get_exchange(indices, integrals)
    (one_rdm_l1, _, _, _, _,
     one_rdm_l2, _, _, _, _) = rdms

    if ct == 'A':
        if localex_l == 'A' and localex_r == 'A':
            h1 = 2 * np.einsum('pqrs, rs, vmlkpqtu, nw->klmntuvw', exchange,
                               one_rdm_l2, three_ct_rdm_l1, zero_ct_rdm_l2)
            h2 = 2 * np.einsum('pqrs, pq, vmlktu, nwrs->klmntuvw', exchange,
                               one_rdm_l1, two_ct_rdm_l1, one_ct_rdm_l2)
        elif localex_l == 'A' and localex_r == 'B':
            h1 = 2 * np.einsum('pqrs, rs, vmlkpq, nwtu->klmntuvw', exchange,
                               one_rdm_l2, two_ct_rdm_l1, one_ct_rdm_l2)
            h2 = 2 * np.einsum('pqrs, pq, vmlk, nwrstu->klmntuvw', exchange,
                               one_rdm_l1, one_ct_rdm_l1, two_ct_rdm_l2)
        elif localex_l == 'B' and localex_r == 'A':
            h1 = 2 * np.einsum('pqrs, rs, vmpqtu, nwlk->klmntuvw', exchange,
                               one_rdm_l2, two_ct_rdm_l1, one_ct_rdm_l2)
            h2 = 2 * np.einsum('pqrs, pq, vmtu, nwlkrs->klmntuvw', exchange,
                               one_rdm_l1, one_ct_rdm_l1, two_ct_rdm_l2)
        elif localex_l == 'B' and localex_r == 'B':
            h1 = 2 * np.einsum('pqrs, rs, vmpq, nwlktu->klmntuvw', exchange,
                               one_rdm_l2, one_ct_rdm_l1, two_ct_rdm_l2)
            h2 = 2 * np.einsum('pqrs, pq, vm, nwlkrstu->klmntuvw', exchange,
                               one_rdm_l1, zero_ct_rdm_l1, three_ct_rdm_l2)
    if ct == 'B':
        if localex_l == 'A' and localex_r == 'A':
            h1 = 2 * np.einsum('pqrs, rs, nwlkpqtu, vm->klmntuvw', exchange,
                               one_rdm_l2, three_ct_rdm_l1, zero_ct_rdm_l2)
            h2 = 2 * np.einsum('pqrs, pq, nwlktu, vmrs->klmntuvw', exchange,
                               one_rdm_l1, two_ct_rdm_l1, one_ct_rdm_l2)
        elif localex_l == 'A' and localex_r == 'B':
            h1 = 2 * np.einsum('pqrs, rs, nwlkpq, vmtu->klmntuvw', exchange,
                               one_rdm_l2, two_ct_rdm_l1, one_ct_rdm_l2)
            h2 = 2 * np.einsum('pqrs, pq, nwlk, vmrstu->klmntuvw', exchange,
                               one_rdm_l1, one_ct_rdm_l1, two_ct_rdm_l2)
        elif localex_l == 'B' and localex_r == 'A':
            h1 = 2 * np.einsum('pqrs, rs, nwpqtu, vmlk->klmntuvw', exchange,
                               one_rdm_l2, two_ct_rdm_l1, one_ct_rdm_l2)
            h2 = 2 * np.einsum('pqrs, pq, nwtu, vmlkrs->klmntuvw', exchange,
                               one_rdm_l1, one_ct_rdm_l1, two_ct_rdm_l2)
        elif localex_l == 'B' and localex_r == 'B':
            h1 = 2 * np.einsum('pqrs, rs, nwpq, vmlktu->klmntuvw', exchange,
                               one_rdm_l2, one_ct_rdm_l1, two_ct_rdm_l2)
            h2 = 2 * np.einsum('pqrs, pq, nw, vmlkrstu->klmntuvw', exchange,
                               one_rdm_l1, zero_ct_rdm_l1, three_ct_rdm_l2)
    h3 = - np.einsum('pqrs, pq, rs', exchange,
                     one_rdm_l1, one_rdm_l2) * get_overlap_ct(
                         ct, localex_l, localex_r, ct_rdms)
    return h1 + h2 + h3


def get_overlap_ct(ct, localex_l, localex_r, ct_rdms):
    (zero_ct_rdm_l1, one_ct_rdm_l1, two_ct_rdm_l1, _, _,
     zero_ct_rdm_l2, one_ct_rdm_l2, two_ct_rdm_l2, _, _) = ct_rdms[ct]
    if ct == 'A':
        if localex_l == 'A' and localex_r == 'A':
            return 2 * np.einsum('vmlktu, nw->klmntuvw', two_ct_rdm_l1, zero_ct_rdm_l2)
        elif localex_l == 'A' and localex_r == 'B':
            return 2 * np.einsum('vmlk, nwtu->klmntuvw', one_ct_rdm_l1, one_ct_rdm_l2)
        elif localex_l == 'B' and localex_r == 'A':
            return 2 * np.einsum('vmtu, nwlk->klmntuvw', one_ct_rdm_l1, one_ct_rdm_l2)
        elif localex_l == 'B' and localex_r == 'B':
            return 2 * np.einsum('vm, nwlktu->klmntuvw', zero_ct_rdm_l1, two_ct_rdm_l2)

    elif ct == 'B':
        if localex_l == 'A' and localex_r == 'A':
            return 2 * np.einsum('nwlktu, vm->klmntuvw', two_ct_rdm_l1, zero_ct_rdm_l2)
        elif localex_l == 'A' and localex_r == 'B':
            return 2 * np.einsum('nwlk, vmtu->klmntuvw', one_ct_rdm_l1, one_ct_rdm_l2)
        elif localex_l == 'B' and localex_r == 'A':
            return 2 * np.einsum('nwtu, vmlk->klmntuvw', one_ct_rdm_l1, one_ct_rdm_l2)
        elif localex_l == 'B' and localex_r == 'B':
            return 2 * np.einsum('nw, vmlktu->klmntuvw', zero_ct_rdm_l1, two_ct_rdm_l2)


def get_ham_ct_col(ct, localex, indices, ct_rdms, integrals):
    l1_idx, l2_idx = indices
    (zero_ct_rdm_l1, one_ct_rdm_l1, two_ct_rdm_l1, _, _,
     zero_ct_rdm_l2, one_ct_rdm_l2, two_ct_rdm_l2, _, _) = ct_rdms[ct]
    c1, c2 = integrals

    g_eff_A = c2[np.ix_(l1_idx, l1_idx, l1_idx, l2_idx)]
    g_eff_B = c2[np.ix_(l2_idx, l2_idx, l2_idx, l1_idx)]

    if ct == 'A':
        h_eff = c1[np.ix_(l1_idx, l2_idx)] - np.einsum(
            'prrq->pq', c2[np.ix_(l1_idx, l1_idx, l1_idx, l2_idx)])
        if localex == 'A':
            h1 = 2 * np.einsum('pq, pmlk, nq->klmn', h_eff,
                               one_ct_rdm_l1, zero_ct_rdm_l2)
            h2_A = 2 * np.einsum('pqrs, rmlkpq, ns->klmn', g_eff_A,
                                 two_ct_rdm_l1, zero_ct_rdm_l2)
            h2_B = 2 * np.einsum('pqrs, smlk, nrpq->klmn', g_eff_B,
                                 one_ct_rdm_l1, one_ct_rdm_l2)
        elif localex == 'B':
            h1 = 2 * np.einsum('pq, pm, nqlk->klmn', h_eff,
                               zero_ct_rdm_l1, one_ct_rdm_l2)
            h2_A = 2 * np.einsum('pqrs, rmpq, nslk->klmn', g_eff_A,
                                 one_ct_rdm_l1, one_ct_rdm_l2)
            h2_B = 2 * np.einsum('pqrs, sm, nrlkpq->klmn', g_eff_B,
                                 zero_ct_rdm_l1, two_ct_rdm_l2)
    elif ct == 'B':
        h_eff = c1[np.ix_(l2_idx, l1_idx)] - np.einsum(
            'prrq->pq', c2[np.ix_(l2_idx, l2_idx, l2_idx, l1_idx)])
        if localex == 'A':
            h1 = 2 * np.einsum('pq, nqlk, pm->klmn', h_eff,
                               one_ct_rdm_l1, zero_ct_rdm_l2)
            h2_A = 2 * np.einsum('pqrs, nrlkpq, sm->klmn', g_eff_A,
                                 two_ct_rdm_l1, zero_ct_rdm_l2)
            h2_B = 2 * np.einsum('pqrs, nslk, rmpq->klmn', g_eff_B,
                                 one_ct_rdm_l1, one_ct_rdm_l2)
        elif localex == 'B':
            h1 = 2 * np.einsum('pq, nq, pmlk->klmn', h_eff,
                               zero_ct_rdm_l1, one_ct_rdm_l2)
            h2_A = 2 * np.einsum('pqrs, nrpq, smlk->klmn', g_eff_A,
                                 one_ct_rdm_l1, one_ct_rdm_l2)
            h2_B = 2 * np.einsum('pqrs, ns, rmlkpq->klmn', g_eff_B,
                                 zero_ct_rdm_l1, two_ct_rdm_l2)
    return h1 + h2_A + h2_B


# def compute_all_ct_matrices(ct):
#     nex_AA = ncas_l1 * ncas_l1 * ncas_l1 * ncas_l2
#     nex_BB = ncas_l2 * ncas_l2 * ncas_l1 * ncas_l2

#     S_ovlp_AA_AA = get_overlap_ct(ct, 'A', 'A').reshape((nex_AA, nex_AA))
#     S_ovlp_AA_BB = get_overlap_ct(ct, 'A', 'B').reshape((nex_AA, nex_BB))
#     S_ovlp_BB_AA = get_overlap_ct(ct, 'B', 'A').reshape((nex_BB, nex_AA))
#     S_ovlp_BB_BB = get_overlap_ct(ct, 'B', 'B').reshape((nex_BB, nex_BB))

#     H_l1_AA_AA = get_h0_pure_ct('l1', ct, 'A', 'A').reshape((nex_AA, nex_AA))
#     H_l1_AA_BB = get_h0_pure_ct('l1', ct, 'A', 'B').reshape((nex_AA, nex_BB))
#     H_l1_BB_AA = get_h0_pure_ct('l1', ct, 'B', 'A').reshape((nex_BB, nex_AA))
#     H_l1_BB_BB = get_h0_pure_ct('l1', ct, 'B', 'B').reshape((nex_BB, nex_BB))

#     H_l2_AA_AA = get_h0_pure_ct('l2', ct, 'A', 'A').reshape((nex_AA, nex_AA))
#     H_l2_AA_BB = get_h0_pure_ct('l2', ct, 'A', 'B').reshape((nex_AA, nex_BB))
#     H_l2_BB_AA = get_h0_pure_ct('l2', ct, 'B', 'A').reshape((nex_BB, nex_AA))
#     H_l2_BB_BB = get_h0_pure_ct('l2', ct, 'B', 'B').reshape((nex_BB, nex_BB))

#     H_0_ct_AA_AA = get_h0_disp_ct(ct, 'A', 'A').reshape((nex_AA, nex_AA))
#     H_0_ct_AA_BB = get_h0_disp_ct(ct, 'A', 'B').reshape((nex_AA, nex_BB))
#     H_0_ct_BB_AA = get_h0_disp_ct(ct, 'B', 'A').reshape((nex_BB, nex_AA))
#     H_0_ct_BB_BB = get_h0_disp_ct(ct, 'B', 'B').reshape((nex_BB, nex_BB))

#     S_ovlp = np.block([[S_ovlp_AA_AA, S_ovlp_AA_BB],
#                        [S_ovlp_BB_AA, S_ovlp_BB_BB]])
#     H_l1 = np.block([[H_l1_AA_AA, H_l1_AA_BB],
#                      [H_l1_BB_AA, H_l1_BB_BB]])
#     H_l2 = np.block([[H_l2_AA_AA, H_l2_AA_BB],
#                      [H_l2_BB_AA, H_l2_BB_BB]])
#     H_0_ct = np.block([[H_0_ct_AA_AA, H_0_ct_AA_BB],
#                        [H_0_ct_BB_AA, H_0_ct_BB_BB]])

#     H_0 = H_l1 + H_l2 + H_0_ct

#     # print(f'Hermitian check ct={ct}, should be true:')
#     # print(np.allclose(S_ovlp, S_ovlp.T))
#     # print(np.allclose(H_l1, H_l1.T))
#     # print(np.allclose(H_l2, H_l2.T))
#     # print(np.allclose(H_0_ct, H_0_ct.T))
#     # print(np.allclose(H_0, H_0.T))

#     return S_ovlp, H_0


# def run_fragpt2_ct():
#     get_four_rdms()
#     get_five_rdms()

#     E_0 = h0_expval(w_core=False)

#     # -------- AB ----------
#     S_ovlp_AB, H_0_AB = compute_all_ct_matrices('A')

#     nex_AAAB = ncas_l1 * ncas_l1 * ncas_l1 * ncas_l2
#     nex_BBAB = ncas_l2 * ncas_l2 * ncas_l1 * ncas_l2

#     H_prime_col_AAAB = get_ham_ct_col('A', 'A')
#     H_prime_col_AAAB = H_prime_col_AAAB.reshape((nex_AAAB))
#     H_prime_col_BBAB = get_ham_ct_col('A', 'B')
#     H_prime_col_BBAB = H_prime_col_BBAB.reshape((nex_BBAB))
#     H_prime_col_AB = np.concatenate((H_prime_col_AAAB, H_prime_col_BBAB))

#     psi1_c_ct_AB = np.linalg.solve((H_0_AB - E_0 * S_ovlp_AB),
#                                    - H_prime_col_AB)
#     e_pt2_AB = np.einsum('n, n', H_prime_col_AB,
#                          psi1_c_ct_AB)

#     # -------- BA ----------
#     S_ovlp_BA, H_0_BA = compute_all_ct_matrices('B')

#     nex_AABA = ncas_l1 * ncas_l1 * ncas_l2 * ncas_l1
#     nex_BBBA = ncas_l2 * ncas_l2 * ncas_l2 * ncas_l1

#     H_prime_col_AABA = get_ham_ct_col('B', 'A')
#     H_prime_col_AABA = H_prime_col_AABA.reshape((nex_AABA))
#     H_prime_col_BBBA = get_ham_ct_col('B', 'B')
#     H_prime_col_BBBA = H_prime_col_BBBA.reshape((nex_BBBA))
#     H_prime_col_BA = np.concatenate((H_prime_col_AABA, H_prime_col_BBBA))

#     psi1_c_ct_BA = np.linalg.solve((H_0_BA - E_0 * S_ovlp_BA),
#                                    - H_prime_col_BA)

#     e_pt2_BA = np.einsum('n, n', H_prime_col_BA,
#                          psi1_c_ct_BA)

#     return e_pt2_AB, e_pt2_BA
