#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Created on Tue Apr 16 12:24:45 2024

@author: emielkoridon
Functions that need to be defined in every module in ../perturbations:
    - get_h0
    - get_ovlp
    - get_h_prime


Computes second-order perturbation energy for dispersion interactions.
Shares k-RDMs up to k=4.

Uses the perturbing functions:

.. math::
    \{ |\Psi_{tuvw} \rangle = E_{t_A u_B} E_{v_A w_B} |\Psi^0\rangle \}

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


def get_h0(ct, indices, rdms, ct2_rdms, integrals):
    nex = get_nex(indices)
    H_l1 = get_h0_pure_2ct('l1', ct, indices, rdms, ct2_rdms, integrals)
    H_l2 = get_h0_pure_2ct('l2', ct, indices, rdms, ct2_rdms, integrals)
    H_0_disp = get_h0_disp_2ct(ct, indices, rdms, ct2_rdms, integrals)
    H_0 = H_l1 + H_l2 + H_0_disp
    H_0 = H_0.reshape((nex, nex))
    return H_0


def get_ovlp(ct, indices, ct2_rdms):
    nex = get_nex(indices)
    S_ovlp = contract_2ct_mel(None, (0, 0), ct, ct2_rdms)
    return S_ovlp.reshape((nex, nex))


def get_h_prime(ct, indices, ct2_rdms, integrals):
    nex = get_nex(indices)
    H_prime_col = get_h_prime_2ct(ct, indices, ct2_rdms, integrals)
    H_prime_col = H_prime_col.reshape((nex))
    return H_prime_col


# UTILITY FUNCTIONS

def contract_2ct_mel(tensor, order, ct, ct2_rdms):
    oa, ob = order
    (ct2_rdms_abba_l1, ct2_rdms_aaaa_l1, ct2_rdms_bbbb_l1,
     ct2_rdms_abba_l2, ct2_rdms_aaaa_l2, ct2_rdms_bbbb_l2) = ct2_rdms[ct]
    full = 'pqrs'
    # import pdb
    # pdb.set_trace()
    if oa == 0 and ob == 0:
        if ct == 'A':
            return sum((
                np.einsum('mktv, nluw->klmntuvw',
                          ct2_rdms_aaaa_l1[oa//2], ct2_rdms_aaaa_l2[ob//2]),
                np.einsum('mktv, nluw->klmntuvw',
                          ct2_rdms_bbbb_l1[oa//2], ct2_rdms_bbbb_l2[ob//2]),
                np.einsum('mktv, nluw->klmntuvw',
                          ct2_rdms_abba_l1[oa//2], ct2_rdms_abba_l2[ob//2]),
                np.einsum('mkvt, nlwu->klmntuvw',
                          ct2_rdms_abba_l1[oa//2], ct2_rdms_abba_l2[ob//2]),
                np.einsum('kmtv, lnuw->klmntuvw',
                          ct2_rdms_abba_l1[oa//2], ct2_rdms_abba_l2[ob//2]),
                np.einsum('kmvt, lnwu->klmntuvw',
                          ct2_rdms_abba_l1[oa//2], ct2_rdms_abba_l2[ob//2])
            ))
        elif ct == 'B':
            return sum((
                np.einsum('nluw, mktv->klmntuvw',
                          ct2_rdms_aaaa_l1[oa//2], ct2_rdms_aaaa_l2[ob//2]),
                np.einsum('nluw, mktv->klmntuvw',
                          ct2_rdms_bbbb_l1[oa//2], ct2_rdms_bbbb_l2[ob//2]),
                np.einsum('nluw, mktv->klmntuvw',
                          ct2_rdms_abba_l1[oa//2], ct2_rdms_abba_l2[ob//2]),
                np.einsum('nlwu, mkvt->klmntuvw',
                          ct2_rdms_abba_l1[oa//2], ct2_rdms_abba_l2[ob//2]),
                np.einsum('lnuw, kmtv->klmntuvw',
                          ct2_rdms_abba_l1[oa//2], ct2_rdms_abba_l2[ob//2]),
                np.einsum('lnwu, kmvt->klmntuvw',
                          ct2_rdms_abba_l1[oa//2], ct2_rdms_abba_l2[ob//2])
            ))
    else:
        if ct == 'A':
            return sum((
                np.einsum(f'{full[:oa+ob]}, mk{full[:oa]}tv, nl{full[oa:oa+ob]}uw->klmntuvw',
                          tensor, ct2_rdms_aaaa_l1[oa//2], ct2_rdms_aaaa_l2[ob//2]),
                np.einsum(f'{full[:oa+ob]}, mk{full[:oa]}tv, nl{full[oa:oa+ob]}uw->klmntuvw',
                          tensor, ct2_rdms_bbbb_l1[oa//2], ct2_rdms_bbbb_l2[ob//2]),
                np.einsum(f'{full[:oa+ob]}, mk{full[:oa]}tv, nl{full[oa:oa+ob]}uw->klmntuvw',
                          tensor, ct2_rdms_abba_l1[oa//2], ct2_rdms_abba_l2[ob//2]),
                np.einsum(f'{full[:oa+ob]}, mk{full[:oa]}vt, nl{full[oa:oa+ob]}wu->klmntuvw',
                          tensor, ct2_rdms_abba_l1[oa//2], ct2_rdms_abba_l2[ob//2]),
                np.einsum(f'{full[:oa+ob]}, km{full[:oa]}tv, ln{full[oa:oa+ob]}uw->klmntuvw',
                          tensor, ct2_rdms_abba_l1[oa//2], ct2_rdms_abba_l2[ob//2]),
                np.einsum(f'{full[:oa+ob]}, km{full[:oa]}vt, ln{full[oa:oa+ob]}wu->klmntuvw',
                          tensor, ct2_rdms_abba_l1[oa//2], ct2_rdms_abba_l2[ob//2])
            ))
        elif ct == 'B':
            return sum((
                np.einsum(f'{full[:oa+ob]}, nl{full[:oa]}uw, mk{full[oa:oa+ob]}tv->klmntuvw',
                          tensor, ct2_rdms_aaaa_l1[oa//2], ct2_rdms_aaaa_l2[ob//2]),
                np.einsum(f'{full[:oa+ob]}, nl{full[:oa]}uw, mk{full[oa:oa+ob]}tv->klmntuvw',
                          tensor, ct2_rdms_bbbb_l1[oa//2], ct2_rdms_bbbb_l2[ob//2]),
                np.einsum(f'{full[:oa+ob]}, nl{full[:oa]}uw, mk{full[oa:oa+ob]}tv->klmntuvw',
                          tensor, ct2_rdms_abba_l1[oa//2], ct2_rdms_abba_l2[ob//2]),
                np.einsum(f'{full[:oa+ob]}, nl{full[:oa]}wu, mk{full[oa:oa+ob]}vt->klmntuvw',
                          tensor, ct2_rdms_abba_l1[oa//2], ct2_rdms_abba_l2[ob//2]),
                np.einsum(f'{full[:oa+ob]}, ln{full[:oa]}uw, km{full[oa:oa+ob]}tv->klmntuvw',
                          tensor, ct2_rdms_abba_l1[oa//2], ct2_rdms_abba_l2[ob//2]),
                np.einsum(f'{full[:oa+ob]}, ln{full[:oa]}wu, km{full[oa:oa+ob]}vt->klmntuvw',
                          tensor, ct2_rdms_abba_l1[oa//2], ct2_rdms_abba_l2[ob//2])
            ))


def get_h0_pure_2ct(fragment, ct, indices, rdms, ct2_rdms, integrals):
    l1_idx, l2_idx = indices
    c1, c2 = integrals

    if fragment == 'l1':
        c1_pure = c1[np.ix_(*[l1_idx]*2)]
        c2_pure = c2[np.ix_(*[l1_idx]*4)]
        c1_prime = h_prime(c1_pure, c2_pure)

        h1 = contract_2ct_mel(c1_prime, (2, 0), ct, ct2_rdms)
        h2 = .5 * contract_2ct_mel(c2_pure, (4, 0), ct, ct2_rdms)

    elif fragment == 'l2':
        c1_pure = c1[np.ix_(*[l2_idx]*2)]
        c2_pure = c2[np.ix_(*[l2_idx]*4)]
        c1_prime = h_prime(c1_pure, c2_pure)

        h1 = contract_2ct_mel(c1_prime, (0, 2), ct, ct2_rdms)
        h2 = .5 * contract_2ct_mel(c2_pure, (0, 4), ct, ct2_rdms)

    return h1 + h2


def get_h0_disp_2ct(ct, indices, rdms, ct2_rdms, integrals):
    l1_idx, l2_idx = indices
    c1, c2 = integrals
    exchange = get_exchange(indices, integrals)
    (one_rdm_l1, _, _, _,
     one_rdm_l2, _, _, _) = rdms

    h1 = contract_2ct_mel(np.einsum('pqrs, rs->pq', exchange, one_rdm_l2),
                          (2, 0), ct, ct2_rdms)

    h2 = contract_2ct_mel(np.einsum('pqrs, pq->rs', exchange, one_rdm_l1),
                          (0, 2), ct, ct2_rdms)

    h3 = - np.einsum('pqrs, pq, rs', exchange, one_rdm_l1, one_rdm_l2) * contract_2ct_mel(
        None, (0, 0), ct, ct2_rdms)

    return h1 + h2 + h3


def get_h_prime_2ct(ct, indices, ct2_rdms, integrals):
    l1_idx, l2_idx = indices
    c1, c2 = integrals
    if ct == 'A':
        c2_eff = c2[np.ix_(l2_idx, l1_idx, l2_idx, l1_idx)]
    elif ct == 'B':
        c2_eff = c2[np.ix_(l1_idx, l2_idx, l1_idx, l2_idx)]

    return .5 * np.einsum('pqrs, srqptuvw->tuvw', c2_eff,
                          contract_2ct_mel(None, (0, 0), ct, ct2_rdms))


# def get_h0_pure_2ct(fragment, ct, indices, rdms, ct2_rdms, integrals):
#     l1_idx, l2_idx = indices
#     c1, c2 = integrals
#     (ct2_rdms_abba_l1, ct2_rdms_aaaa_l1, ct2_rdms_bbbb_l1,
#      ct2_rdms_abba_l2, ct2_rdms_aaaa_l2, ct2_rdms_bbbb_l2) = ct2_rdms[ct]

#     if fragment == 'l1':
#         c1_pure = c1[np.ix_(*[l1_idx]*2)]
#         c2_pure = c2[np.ix_(*[l1_idx]*4)]
#         c1_prime = h_prime(c1_pure, c2_pure)
#         h1 = sum((
#             np.einsum('pq, mkpqtv, nluw->klmntuvw', c1_prime,
#                       ct2_rdms_aaaa_l1[1], ct2_rdms_aaaa_l2[0]),
#             np.einsum('pq, mkpqtv, nluw->klmntuvw', c1_prime,
#                       ct2_rdms_bbbb_l1[1], ct2_rdms_bbbb_l2[0]),
#             np.einsum('pq, mkpqtv, nluw->klmntuvw', c1_prime,
#                       ct2_rdms_abba_l1[1], ct2_rdms_abba_l2[0]),
#             np.einsum('pq, mkpqvt, nlwu->klmntuvw', c1_prime,
#                       ct2_rdms_abba_l1[1], ct2_rdms_abba_l2[0]),
#             np.einsum('pq, kmpqtv, lnuw->klmntuvw', c1_prime,
#                       ct2_rdms_abba_l1[1], ct2_rdms_abba_l2[0]),
#             np.einsum('pq, kmpqvt, lnwu->klmntuvw', c1_prime,
#                       ct2_rdms_abba_l1[1], ct2_rdms_abba_l2[0])
#         ))

#         h2 = .5 * sum((
#             np.einsum('pqrs, mkpqrstv, nluw->klmntuvw', c2_pure,
#                       ct2_rdms_aaaa_l1[2], ct2_rdms_aaaa_l2[0]),
#             np.einsum('pqrs, mkpqrstv, nluw->klmntuvw', c2_pure,
#                       ct2_rdms_bbbb_l1[2], ct2_rdms_bbbb_l2[0]),
#             np.einsum('pqrs, mkpqrstv, nluw->klmntuvw', c2_pure,
#                       ct2_rdms_abba_l1[2], ct2_rdms_abba_l2[0]),
#             np.einsum('pqrs, mkpqrsvt, nlwu->klmntuvw', c2_pure,
#                       ct2_rdms_abba_l1[2], ct2_rdms_abba_l2[0]),
#             np.einsum('pqrs, kmpqrstv, lnuw->klmntuvw', c2_pure,
#                       ct2_rdms_abba_l1[2], ct2_rdms_abba_l2[0]),
#             np.einsum('pqrs, kmpqrsvt, lnwu->klmntuvw', c2_pure,
#                       ct2_rdms_abba_l1[2], ct2_rdms_abba_l2[0])
#         ))
#     elif fragment == 'l2':
#         c1_pure = c1[np.ix_(*[l2_idx]*2)]
#         c2_pure = c2[np.ix_(*[l2_idx]*4)]
#         c1_prime = h_prime(c1_pure, c2_pure)

#         h1 = sum((
#             np.einsum('pq, mktv, nlpquw->klmntuvw', c1_prime,
#                       ct2_rdms_aaaa_l1[0], ct2_rdms_aaaa_l2[1]),
#             np.einsum('pq, mktv, nlpquw->klmntuvw', c1_prime,
#                       ct2_rdms_bbbb_l1[0], ct2_rdms_bbbb_l2[1]),
#             np.einsum('pq, mktv, nlpquw->klmntuvw', c1_prime,
#                       ct2_rdms_abba_l1[0], ct2_rdms_abba_l2[1]),
#             np.einsum('pq, mkvt, nlpqwu->klmntuvw', c1_prime,
#                       ct2_rdms_abba_l1[0], ct2_rdms_abba_l2[1]),
#             np.einsum('pq, kmtv, lnpquw->klmntuvw', c1_prime,
#                       ct2_rdms_abba_l1[0], ct2_rdms_abba_l2[1]),
#             np.einsum('pq, kmvt, lnpqwu->klmntuvw', c1_prime,
#                       ct2_rdms_abba_l1[0], ct2_rdms_abba_l2[1])
#         ))

#         h2 = .5 * sum((
#             np.einsum('pqrs, mktv, nlpqrsuw->klmntuvw', c2_pure,
#                       ct2_rdms_aaaa_l1[0], ct2_rdms_aaaa_l2[2]),
#             np.einsum('pqrs, mktv, nlpqrsuw->klmntuvw', c2_pure,
#                       ct2_rdms_bbbb_l1[0], ct2_rdms_bbbb_l2[2]),
#             np.einsum('pqrs, mktv, nlpqrsuw->klmntuvw', c2_pure,
#                       ct2_rdms_abba_l1[0], ct2_rdms_abba_l2[2]),
#             np.einsum('pqrs, mkvt, nlpqrswu->klmntuvw', c2_pure,
#                       ct2_rdms_abba_l1[0], ct2_rdms_abba_l2[2]),
#             np.einsum('pqrs, kmtv, lnpqrsuw->klmntuvw', c2_pure,
#                       ct2_rdms_abba_l1[0], ct2_rdms_abba_l2[2]),
#             np.einsum('pqrs, kmvt, lnpqrswu->klmntuvw', c2_pure,
#                       ct2_rdms_abba_l1[0], ct2_rdms_abba_l2[2])
#         ))
#     return h1 + h2


# def get_2ct_mel(order, ct, ct2_rdms):
#     oa, ob = order
#     (ct2_rdms_abba_l1, ct2_rdms_aaaa_l1, ct2_rdms_bbbb_l1,
#      ct2_rdms_abba_l2, ct2_rdms_aaaa_l2, ct2_rdms_bbbb_l2) = ct2_rdms[ct]
#     full = 'pqrs'
#     if ct == 'A':
#         return sum((
#             np.einsum(f'mk{full[:oa]}tv, nl{full[oa:oa+ob]}uw->klmn{full[:oa+ob]}tuvw',
#                       ct2_rdms_aaaa_l1[oa//2], ct2_rdms_aaaa_l2[ob//2]),
#             np.einsum(f'mk{full[:oa]}tv, nl{full[oa:oa+ob]}uw->klmn{full[:oa+ob]}tuvw',
#                       ct2_rdms_bbbb_l1[oa//2], ct2_rdms_bbbb_l2[ob//2]),
#             np.einsum(f'mk{full[:oa]}tv, nl{full[oa:oa+ob]}uw->klmn{full[:oa+ob]}tuvw',
#                       ct2_rdms_abba_l1[oa//2], ct2_rdms_abba_l2[ob//2]),
#             np.einsum(f'mk{full[:oa]}vt, nl{full[oa:oa+ob]}wu->klmn{full[:oa+ob]}tuvw',
#                       ct2_rdms_abba_l1[oa//2], ct2_rdms_abba_l2[ob//2]),
#             np.einsum(f'km{full[:oa]}tv, ln{full[oa:oa+ob]}uw->klmn{full[:oa+ob]}tuvw',
#                       ct2_rdms_abba_l1[oa//2], ct2_rdms_abba_l2[ob//2]),
#             np.einsum(f'km{full[:oa]}vt, ln{full[oa:oa+ob]}wu->klmn{full[:oa+ob]}tuvw',
#                       ct2_rdms_abba_l1[oa//2], ct2_rdms_abba_l2[ob//2])
#         ))
#     elif ct == 'B':
#         return sum((
#             np.einsum(f'nl{full[:oa]}uw, mk{full[oa:oa+ob]}tv->klmn{full[:oa+ob]}tuvw',
#                       ct2_rdms_aaaa_l1[oa//2], ct2_rdms_aaaa_l2[ob//2]),
#             np.einsum(f'nl{full[:oa]}uw, mk{full[oa:oa+ob]}tv->klmn{full[:oa+ob]}tuvw',
#                       ct2_rdms_bbbb_l1[oa//2], ct2_rdms_bbbb_l2[ob//2]),
#             np.einsum(f'nl{full[:oa]}uw, mk{full[oa:oa+ob]}tv->klmn{full[:oa+ob]}tuvw',
#                       ct2_rdms_abba_l1[oa//2], ct2_rdms_abba_l2[ob//2]),
#             np.einsum(f'nl{full[:oa]}wu, mk{full[oa:oa+ob]}vt->klmn{full[:oa+ob]}tuvw',
#                       ct2_rdms_abba_l1[oa//2], ct2_rdms_abba_l2[ob//2]),
#             np.einsum(f'ln{full[:oa]}uw, km{full[oa:oa+ob]}tv->klmn{full[:oa+ob]}tuvw',
#                       ct2_rdms_abba_l1[oa//2], ct2_rdms_abba_l2[ob//2]),
#             np.einsum(f'ln{full[:oa]}wu, km{full[oa:oa+ob]}vt->klmn{full[:oa+ob]}tuvw',
#                       ct2_rdms_abba_l1[oa//2], ct2_rdms_abba_l2[ob//2])
#         ))


# def get_h0_pure_2ct(fragment, ct, indices, rdms, ct2_rdms, integrals):
#     l1_idx, l2_idx = indices
#     c1, c2 = integrals
#     if fragment == 'l1':
#         c1_pure = c1[np.ix_(*[l1_idx]*2)]
#         c2_pure = c2[np.ix_(*[l1_idx]*4)]
#         c1_prime = h_prime(c1_pure, c2_pure)

#         h1 = np.einsum('pq, klmnpqtuvw->klmntuvw', c1_prime,
#                        get_2ct_mel((2, 0), ct, ct2_rdms))
#         h2 = .5 * np.einsum('pqrs, klmnpqrstuvw->klmntuvw', c2_pure,
#                             get_2ct_mel((4, 0), ct, ct2_rdms))
#     elif fragment == 'l2':
#         c1_pure = c1[np.ix_(*[l2_idx]*2)]
#         c2_pure = c2[np.ix_(*[l2_idx]*4)]
#         c1_prime = h_prime(c1_pure, c2_pure)

#         h1 = np.einsum('pq, klmnpqtuvw->klmntuvw', c1_prime,
#                        get_2ct_mel((0, 2), ct, ct2_rdms))
#         h2 = .5 * np.einsum('pqrs, klmnpqrstuvw->klmntuvw', c2_pure,
#                             get_2ct_mel((0, 4), ct, ct2_rdms))
#     return h1 + h2


# def get_h0_disp_2ct(ct, indices, rdms, ct2_rdms, integrals):
#     l1_idx, l2_idx = indices
#     c1, c2 = integrals
#     exchange = get_exchange(indices, integrals)
#     (one_rdm_l1, _, _, _,
#      one_rdm_l2, _, _, _) = rdms
#     h1 = np.einsum('pqrs, rs, klmnpqtuvw->klmntuvw', exchange,
#                    one_rdm_l2, get_2ct_mel((2, 0), ct, ct2_rdms))

#     h2 = np.einsum('pqrs, pq, klmnrstuvw->klmntuvw', exchange,
#                    one_rdm_l1, get_2ct_mel((0, 2), ct, ct2_rdms))

#     h3 = - np.einsum('pqrs, pq, rs', exchange,
#                      one_rdm_l1, one_rdm_l2) * get_2ct_mel((0, 0), ct, ct2_rdms)
#     return h1 + h2 + h3


# def get_h_prime_2ct(ct, indices, ct2_rdms, integrals):
#     l1_idx, l2_idx = indices
#     c1, c2 = integrals
#     if ct == 'A':
#         c2_eff = c2[np.ix_(l2_idx, l1_idx, l2_idx, l1_idx)]
#     elif ct == 'B':
#         c2_eff = c2[np.ix_(l1_idx, l2_idx, l1_idx, l2_idx)]

#     # import pdb
#     # pdb.set_trace()

#     return .5 * np.einsum('pqrs, srqptuvw->tuvw', c2_eff, get_2ct_mel((0, 0), ct, ct2_rdms))
#     # elif ct == 'B':
#     #     return .5 * np.einsum('pqrs, pqrstuvw->tuvw', c2_eff,
#     #                           np.einsum('rspqtuvw->pqrstuvw', get_2ct_mel((0, 0), ct, ct2_rdms)))
