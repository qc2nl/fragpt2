#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Created on Tue Apr 16 12:25:10 2024

@author: emielkoridon
Functions that need to be defined in every module in ../perturbations:
    - get_h0
    - get_ovlp
    - get_h_prime


Computes second-order perturbation energy for TT interactions.
Needs k-RDMs up to k=4 in spin-orbital basis.

Uses the perturbing functions:

.. math::
    \{ |\Psi_{tuvw} \rangle = TT_{t_A u_A} TT_{v_B w_B} |\Psi^0\rangle \}

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
    return ncas_l1 * ncas_l1 * ncas_l2 * ncas_l2


# MAIN FUNCTIONS

def get_h0(indices, rdms, tt_rdms, integrals):
    (one_rdm_l1, _, _, _,
     one_rdm_l2, _, _, _,) = rdms
    nex = get_nex(indices)
    H_0 = get_h0_tt(indices, one_rdm_l1, one_rdm_l2, tt_rdms, integrals)
    H_0 = H_0.reshape((nex, nex))
    return H_0


def get_ovlp(indices, tt_rdms):
    nex = get_nex(indices)
    S_ovlp = contract_tt_mel(None, (0, 0), tt_rdms).reshape((nex, nex))
    return S_ovlp


def get_h_prime(indices, tt_rdms, integrals):
    nex = get_nex(indices)
    H_prime_col = get_h_tt_col(indices, tt_rdms, integrals)
    H_prime_col = H_prime_col.reshape((nex))
    return H_prime_col


def contract_tt_mel(tensor, order, tt_rdms):
    oa, ob = order
    tt_rdm_abba_l1, tt_rdm_baab_l1, tt_rdm_aabb_l1 = tt_rdms['l1']
    tt_rdm_abba_l2, tt_rdm_baab_l2, tt_rdm_aabb_l2 = tt_rdms['l2']

    full = 'pqrs'
    # import pdb
    # pdb.set_trace()
    if oa == 0 and ob == 0:
        return sum((
            np.einsum('lktu, nmvw -> klmntuvw',
                      tt_rdm_aabb_l1[0], tt_rdm_aabb_l2[0]),
            np.einsum('lktu, nmvw -> klmntuvw',
                      tt_rdm_abba_l1[0], tt_rdm_baab_l2[0]),
            np.einsum('lktu, nmvw -> klmntuvw',
                      tt_rdm_baab_l1[0], tt_rdm_abba_l2[0])
        ))
    else:
        return sum((
            np.einsum(f'{full[:oa+ob]}, lk{full[:oa]}tu, nm{full[oa:oa+ob]}vw -> klmntuvw',
                      tensor, tt_rdm_aabb_l1[oa//2], tt_rdm_aabb_l2[ob//2]),
            np.einsum(f'{full[:oa+ob]}, lk{full[:oa]}tu, nm{full[oa:oa+ob]}vw -> klmntuvw',
                      tensor, tt_rdm_abba_l1[oa//2], tt_rdm_baab_l2[ob//2]),
            np.einsum(f'{full[:oa+ob]}, lk{full[:oa]}tu, nm{full[oa:oa+ob]}vw -> klmntuvw',
                      tensor, tt_rdm_baab_l1[oa//2], tt_rdm_abba_l2[ob//2]),
        ))

# def two_tt_rdm(fragment, tt_rdms):
#     r"""
#     For fragment A:

#     .. math::
#         \langle \Psi_A | T^{(1,0)}_{l k} E_{pq} E_{rs} T^{(1,0)}_{tu} | \Psi_A \rangle
#         \langle \Psi_B | T^{(1,0)}_{n m}  T^{(1,0)}_{vw} | \Psi_B \rangle  \\
#         + \langle \Psi_A | {T^{(1,1)}_{l k} E_{pq} E_{rs} T^{(1,-1)}_{tu} | \Psi_A \rangle
#         \langle \Psi_B | T^{(1,-1)}_{n m} T^{(1,1)}_{vw} | \Psi_B} \rangle \\
#         + \langle \Psi_A | T^{(1,-1)}_{l k} E_{pq} E_{rs} T^{(1,1)}_{tu} | \Psi_A \rangle
#         \langle \Psi_B | T^{(1,1)}_{n m} T^{(1,-1)}_{vw} | \Psi_B \rangle \bigg)

#     for fragment B:

#     .. math::
#         \langle \Psi_A | T^{(1,0)}_{l k}  T^{(1,0)}_{tu} | \Psi_A \rangle
#         \langle \Psi_B | T^{(1,0)}_{n m} E_{pq} E_{rs} T^{(1,0)}_{vw} | \Psi_B \rangle\\
#         + \langle \Psi_A | {T^{(1,1)}_{l k}  T^{(1,-1)}_{tu} | \Psi_A \rangle
#         \langle \Psi_B | T^{(1,-1)}_{n m} E_{pq} E_{rs} T^{(1,1)}_{vw} | \Psi_B} \rangle\\
#         + \langle \Psi_A | T^{(1,-1)}_{l k}  T^{(1,1)}_{tu} | \Psi_A \rangle
#         \langle \Psi_B | T^{(1,1)}_{n m} E_{pq} E_{rs} T^{(1,-1)}_{vw} | \Psi_B \rangle \bigg)

#     """
#     (two_rdm_abba_l1, two_rdm_baab_l1, two_rdm_aabb_l1,
#      three_rdm_abba_l1, three_rdm_baab_l1, three_rdm_aabb_l1,
#      four_rdm_abba_l1, four_rdm_baab_l1, four_rdm_aabb_l1,
#      two_rdm_abba_l2, two_rdm_baab_l2, two_rdm_aabb_l2,
#      three_rdm_abba_l2, three_rdm_baab_l2, three_rdm_aabb_l2,
#      four_rdm_abba_l2, four_rdm_baab_l2, four_rdm_aabb_l2) = tt_rdms
#     if fragment == 'l1':
#         return sum((
#             np.einsum('lkpqrstu, nmvw -> klmnpqrstuvw',
#                       four_rdm_aabb_l1, two_rdm_aabb_l2),
#             np.einsum('lkpqrstu, nmvw -> klmnpqrstuvw',
#                       four_rdm_abba_l1, two_rdm_baab_l2),
#             np.einsum('lkpqrstu, nmvw -> klmnpqrstuvw',
#                       four_rdm_baab_l1, two_rdm_abba_l2)
#         ))
#     elif fragment == 'l2':
#         return sum((
#             np.einsum('lktu, nmpqrsvw -> klmnpqrstuvw',
#                       two_rdm_aabb_l1, four_rdm_aabb_l2),
#             np.einsum('lktu, nmpqrsvw -> klmnpqrstuvw',
#                       two_rdm_abba_l1, four_rdm_baab_l2),
#             np.einsum('lktu, nmpqrsvw -> klmnpqrstuvw',
#                       two_rdm_baab_l1, four_rdm_abba_l2)
#         ))


# def one_tt_rdm(fragment, tt_rdms):
#     r"""
#     For fragment A:

#     .. math::
#         \langle \Psi_A | T^{(1,0)}_{l k} E_{pq} T^{(1,0)}_{tu} | \Psi_A \rangle
#         \langle \Psi_B | T^{(1,0)}_{n m}  T^{(1,0)}_{vw} | \Psi_B \rangle\\
#         + \langle \Psi_A | {T^{(1,1)}_{l k} E_{pq} T^{(1,-1)}_{tu} | \Psi_A \rangle
#         \langle \Psi_B | T^{(1,-1)}_{n m} T^{(1,1)}_{vw} | \Psi_B} \rangle\\
#         + \langle \Psi_A | T^{(1,-1)}_{l k} E_{pq} T^{(1,1)}_{tu} | \Psi_A \rangle
#         \langle \Psi_B | T^{(1,1)}_{n m} T^{(1,-1)}_{vw} | \Psi_B \rangle \bigg)

#     for fragment B:

#     .. math::
#         \langle \Psi_A | T^{(1,0)}_{l k}  T^{(1,0)}_{tu} | \Psi_A \rangle
#         \langle \Psi_B | T^{(1,0)}_{n m} E_{pq} T^{(1,0)}_{vw} | \Psi_B \rangle\\
#         + \langle \Psi_A | {T^{(1,1)}_{l k}  T^{(1,-1)}_{tu} | \Psi_A \rangle
#         \langle \Psi_B | T^{(1,-1)}_{n m} E_{pq} T^{(1,1)}_{vw} | \Psi_B} \rangle\\
#         + \langle \Psi_A | T^{(1,-1)}_{l k}  T^{(1,1)}_{tu} | \Psi_A \rangle
#         \langle \Psi_B | T^{(1,1)}_{n m} E_{pq} T^{(1,-1)}_{vw} | \Psi_B \rangle \bigg)

#     """
#     (two_rdm_abba_l1, two_rdm_baab_l1, two_rdm_aabb_l1,
#      three_rdm_abba_l1, three_rdm_baab_l1, three_rdm_aabb_l1,
#      _, _, _,
#      two_rdm_abba_l2, two_rdm_baab_l2, two_rdm_aabb_l2,
#      three_rdm_abba_l2, three_rdm_baab_l2, three_rdm_aabb_l2,
#      _, _, _) = tt_rdms
#     if fragment == 'l1':
#         return sum((
#             np.einsum('lkpqtu, nmvw -> klmnpqtuvw',
#                       three_rdm_aabb_l1, two_rdm_aabb_l2),
#             np.einsum('lkpqtu, nmvw -> klmnpqtuvw',
#                       three_rdm_abba_l1, two_rdm_baab_l2),
#             np.einsum('lkpqtu, nmvw -> klmnpqtuvw',
#                       three_rdm_baab_l1, two_rdm_abba_l2)
#         ))
#     elif fragment == 'l2':
#         return sum((
#             np.einsum('lktu, nmpqvw -> klmnpqtuvw',
#                       two_rdm_aabb_l1, three_rdm_aabb_l2),
#             np.einsum('lktu, nmpqvw -> klmnpqtuvw',
#                       two_rdm_abba_l1, three_rdm_baab_l2),
#             np.einsum('lktu, nmpqvw -> klmnpqtuvw',
#                       two_rdm_baab_l1, three_rdm_abba_l2)
#         ))


# def zero_tt_rdm(tt_rdms):
#     r"""

#     .. math::
#         \langle \Psi_A | T^{(1,0)}_{l k} T^{(1,0)}_{tu} | \Psi_A \rangle
#         \langle \Psi_B | T^{(1,0)}_{n m}  T^{(1,0)}_{vw} | \Psi_B \rangle\\
#         + \langle \Psi_A | {T^{(1,1)}_{l k} T^{(1,-1)}_{tu} | \Psi_A \rangle
#         \langle \Psi_B | T^{(1,-1)}_{n m} T^{(1,1)}_{vw} | \Psi_B} \rangle\\
#         + \langle \Psi_A | T^{(1,-1)}_{l k} T^{(1,1)}_{tu} | \Psi_A \rangle
#         \langle \Psi_B | T^{(1,1)}_{n m} T^{(1,-1)}_{vw} | \Psi_B \rangle \bigg)
#     """
#     (two_rdm_abba_l1, two_rdm_baab_l1, two_rdm_aabb_l1,
#      _, _, _, _, _, _,
#      two_rdm_abba_l2, two_rdm_baab_l2, two_rdm_aabb_l2,
#      _, _, _, _, _, _) = tt_rdms
#     return sum((
#         np.einsum('lktu, nmvw -> klmntuvw',
#                   two_rdm_aabb_l1, two_rdm_aabb_l2),
#         np.einsum('lktu, nmvw -> klmntuvw',
#                   two_rdm_abba_l1, two_rdm_baab_l2),
#         np.einsum('lktu, nmvw -> klmntuvw',
#                   two_rdm_baab_l1, two_rdm_abba_l2)
#     ))


def get_h0_tt(indices, one_rdm_l1, one_rdm_l2, tt_rdms, integrals):
    l1_idx, l2_idx = indices
    c1, c2 = integrals
    exchange = get_exchange(indices, integrals)

    c1_pure_l1 = c1[np.ix_(*[l1_idx]*2)]
    c2_pure_l1 = c2[np.ix_(*[l1_idx]*4)]

    c1_pure_l2 = c1[np.ix_(*[l2_idx]*2)]
    c2_pure_l2 = c2[np.ix_(*[l2_idx]*4)]

    c1_prime_l1 = h_prime(c1_pure_l1, c2_pure_l1)
    c1_prime_l2 = h_prime(c1_pure_l2, c2_pure_l2)

    h0_pure_one_l1 = contract_tt_mel(c1_prime_l1, (2, 0), tt_rdms)
    h0_pure_two_l1 = .5 * contract_tt_mel(c2_pure_l1, (4, 0), tt_rdms)
    # h0_pure_two_l1 = .5 * np.einsum('pqrs, klmnpqrstuvw->klmntuvw',
    #                                 c2_pure_l1, two_tt_rdm('l1', tt_rdms))

    h0_pure_one_l2 = contract_tt_mel(c1_prime_l2, (0, 2), tt_rdms)
    h0_pure_two_l2 = .5 * contract_tt_mel(c2_pure_l2, (0, 4), tt_rdms)
    # h0_pure_one_l2 = np.einsum('pq, klmnpqtuvw->klmntuvw', c1_prime_l2, one_tt_rdm('l2', tt_rdms))
    # h0_pure_two_l2 = .5 * np.einsum('pqrs, klmnpqrstuvw->klmntuvw',
    #                                 c2_pure_l2, two_tt_rdm('l2', tt_rdms))

    h0_disp_0 = contract_tt_mel(np.einsum(
        'pqrs, rs->pq', exchange, one_rdm_l2), (2, 0), tt_rdms)
    h0_disp_1 = contract_tt_mel(np.einsum('pqrs, pq->rs', exchange,
                                          one_rdm_l1), (0, 2), tt_rdms)

    # h0_disp_0 = np.einsum('pqrs, rs, klmnpqtuvw->klmntuvw', exchange,
    #                       one_rdm_l2, one_tt_rdm('l1', tt_rdms))
    # h0_disp_1 = np.einsum('pqrs, pq, klmnrstuvw->klmntuvw', exchange,
    #                       one_rdm_l1, one_tt_rdm('l2', tt_rdms))

    c_eff = np.einsum('pqrs,pqrs', exchange,
                      np.einsum('pq,rs->pqrs', one_rdm_l1, one_rdm_l2))
    h0_disp_2 = - c_eff * contract_tt_mel(None, (0, 0), tt_rdms)

    return sum((
        h0_pure_one_l1, h0_pure_two_l1, h0_pure_one_l2, h0_pure_two_l2,
        h0_disp_0, h0_disp_1, h0_disp_2))


def get_h_tt_col(indices, tt_rdms, integrals):
    l1_idx, l2_idx = indices
    c1, c2 = integrals
    g_eff = c2[np.ix_(l1_idx, l2_idx, l2_idx, l1_idx)]
    return - np.einsum('psrq, pqrstuvw->tuvw', g_eff, contract_tt_mel(None, (0, 0), tt_rdms))


# def run_fragpt2_tt(self):
#     t1 = time.time()
#     get_tt_rdms()
#     print("computing TT rdms took:", time.time()-t1)
#     E_0 = h0_expval(w_core=False)
#     S_ovlp = zero_tt_rdm().reshape((nex, nex))
#     H_0 = get_h0_tt()
#     H_0 = H_0.reshape((nex, nex))

#     H_prime_col = get_h_tt_col()

#     H_prime_col = H_prime_col.reshape((nex))

#     psi1_c_tt = np.linalg.solve((H_0 - E_0 * S_ovlp),
#                                 -H_prime_col)

#     e_pt2 = np.einsum('n, n', H_prime_col,
#                       psi1_c_tt)
#     return e_pt2
