#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 17:00:06 2024

@author: emielkoridon
"""


from itertools import product
import numpy as np

from pyscf.fci.direct_spin1 import make_rdm12s, trans_rdm1s, trans_rdm12s
from pyscf.fci.addons import cre_a, cre_b, des_a, des_b
from pyscf.fci.rdm import make_dm1234


def compute_five_rdm(civec, norb, nelec):
    r"""
    Construct the five-rdm:

    .. math::
        \langle \Psi | E_{pq} E_{rs} E_{kl} E_{mn} E_{tu} | \Psi \rangle

    """
    five_rdm = np.zeros([norb]*10)
    nelec_a = nelec_b = nelec // 2

    for p in range(norb):
        for q in range(norb):
            civec_desa = des_a(civec, norb, (nelec_a, nelec_b), q)
            civec_credesa = cre_a(civec_desa, norb, (nelec_a-1, nelec_b), p)
            _, _, _, four_rdm_credesa = make_dm1234(
                'FCI4pdm_kern_sf', civec, civec_credesa, norb, nelec)
            civec_desb = des_b(civec, norb, (nelec_a, nelec_b), q)
            civec_credesb = cre_b(civec_desb, norb, (nelec_a, nelec_b-1), p)
            _, _, _, four_rdm_credesb = make_dm1234(
                'FCI4pdm_kern_sf', civec, civec_credesb, norb, nelec)
            four_rdm_spinfree = four_rdm_credesa + four_rdm_credesb
            five_rdm[:, :, :, :, :, :, :, :, p, q] = four_rdm_spinfree
    return five_rdm


def get_1ct_rdms(ct, rdms):
    (one_rdm_l1, two_rdm_l1, three_rdm_l1, four_rdm_l1, five_rdm_l1,
     one_rdm_l2, two_rdm_l2, three_rdm_l2, four_rdm_l2, five_rdm_l2) = rdms
    if ct == 'A':
        zero_ct_rdm_l1 = zero_ct2_rdm(one_rdm_l1)
        one_ct_rdm_l1 = one_ct2_rdm(one_rdm_l1, two_rdm_l1)
        two_ct_rdm_l1 = two_ct2_rdm(one_rdm_l1, two_rdm_l1, three_rdm_l1)
        three_ct_rdm_l1 = three_ct2_rdm(one_rdm_l1, two_rdm_l1, three_rdm_l1, four_rdm_l1)
        four_ct_rdm_l1 = four_ct2_rdm(
            one_rdm_l1, two_rdm_l1, three_rdm_l1, four_rdm_l1, five_rdm_l1)

        zero_ct_rdm_l2 = .5 * one_rdm_l2
        one_ct_rdm_l2 = one_ct1_rdm(one_rdm_l2, two_rdm_l2)
        two_ct_rdm_l2 = two_ct1_rdm(one_rdm_l2, two_rdm_l2, three_rdm_l2)
        three_ct_rdm_l2 = three_ct1_rdm(one_rdm_l2, two_rdm_l2, three_rdm_l2, four_rdm_l2)
        four_ct_rdm_l2 = four_ct1_rdm(
            one_rdm_l2, two_rdm_l2, three_rdm_l2, four_rdm_l2, five_rdm_l2)
    elif ct == 'B':
        zero_ct_rdm_l1 = .5 * one_rdm_l1
        one_ct_rdm_l1 = one_ct1_rdm(one_rdm_l1, two_rdm_l1)
        two_ct_rdm_l1 = two_ct1_rdm(one_rdm_l1, two_rdm_l1, three_rdm_l1)
        three_ct_rdm_l1 = three_ct1_rdm(one_rdm_l1, two_rdm_l1, three_rdm_l1, four_rdm_l1)
        four_ct_rdm_l1 = four_ct1_rdm(
            one_rdm_l1, two_rdm_l1, three_rdm_l1, four_rdm_l1, five_rdm_l1)

        zero_ct_rdm_l2 = zero_ct2_rdm(one_rdm_l2)
        one_ct_rdm_l2 = one_ct2_rdm(one_rdm_l2, two_rdm_l2)
        two_ct_rdm_l2 = two_ct2_rdm(one_rdm_l2, two_rdm_l2, three_rdm_l2)
        three_ct_rdm_l2 = three_ct2_rdm(one_rdm_l2, two_rdm_l2, three_rdm_l2, four_rdm_l2)
        four_ct_rdm_l2 = four_ct2_rdm(
            one_rdm_l2, two_rdm_l2, three_rdm_l2, four_rdm_l2, five_rdm_l2)

    return (zero_ct_rdm_l1, one_ct_rdm_l1, two_ct_rdm_l1, three_ct_rdm_l1, four_ct_rdm_l1,
            zero_ct_rdm_l2, one_ct_rdm_l2, two_ct_rdm_l2, three_ct_rdm_l2, four_ct_rdm_l2)


def zero_ct2_rdm(one_rdm):
    r"""
    .. math::
        \langle a_{t\sigma} a_{k\sigma}^\dagger \rangle
    """
    ncas = one_rdm.shape[0]
    return - .5 * one_rdm + np.eye(ncas)


def one_ct1_rdm(one_rdm, two_rdm):
    r"""
    .. math::
        \langle a_{l\sigma}^\dagger E_{pq} a_{u\sigma} \rangle
    """
    ncas = one_rdm.shape[0]
    return .5 * (two_rdm - np.einsum(
        'pu, lq->lupq', np.eye(ncas), one_rdm))


def one_ct2_rdm(one_rdm, two_rdm):
    r"""
    .. math::
        \langle a_{k\sigma} E_{pq} a_{t\sigma}^\dagger \rangle
    """
    ncas = one_rdm.shape[0]
    return -.5 * two_rdm + np.einsum(
        'kt, pq->tkpq', np.eye(ncas), one_rdm) - .5 * np.einsum(
            'qt, pk->tkpq', np.eye(ncas), one_rdm) + np.einsum(
                'pk, qt->tkpq', np.eye(ncas), np.eye(ncas))


def two_ct1_rdm(one_rdm, two_rdm, three_rdm):
    r"""
    .. math::
        \langle a_{l\sigma}^\dagger E_{pq} E_{rs} a_{u\sigma} \rangle
    """
    ncas = one_rdm.shape[0]
    return .5 * three_rdm - .5 * np.einsum(
        'pu, lqrs->lupqrs', np.eye(ncas), two_rdm) - .5 * np.einsum(
            'ru, lspq->lupqrs', np.eye(ncas), two_rdm) + .5 * np.einsum(
                'ru, ps, lq->lupqrs', np.eye(ncas), np.eye(ncas),
                one_rdm)


def two_ct2_rdm(one_rdm, two_rdm, three_rdm):
    r"""
    .. math::
        \langle a_{k\sigma} E_{pq} E_{rs} a_{t\sigma}^\dagger \rangle
    """
    ncas = one_rdm.shape[0]
    return sum((
        - .5 * three_rdm,
        np.einsum('tk, pqrs->tkpqrs', np.eye(ncas), two_rdm),
        - .5 * np.einsum('tq, pkrs->tkpqrs', np.eye(ncas), two_rdm),
        np.einsum('tq, kp, rs->tkpqrs', np.eye(ncas), np.eye(ncas),
                  one_rdm),
        - .5 * np.einsum('ts, rkpq->tkpqrs', np.eye(ncas), two_rdm),
        np.einsum('ts, kr, pq->tkpqrs', np.eye(ncas), np.eye(ncas),
                  one_rdm),
        - .5 * np.einsum('ts, qr, pk->tkpqrs', np.eye(ncas), np.eye(ncas),
                         one_rdm),
        np.einsum('ts, pk, qr->tkpqrs', np.eye(ncas), np.eye(ncas),
                  np.eye(ncas))
    ))


def three_ct1_rdm(one_rdm, two_rdm, three_rdm, four_rdm):
    r"""
    .. math::
        \langle a_{n\sigma}^\dagger E_{lk} E_{pq} E_{rs} a_{u\sigma} \rangle
    """
    ncas = one_rdm.shape[0]
    return sum((
        .5 * four_rdm,
        - np.einsum('ru, nslkpq->nulkpqrs', np.eye(ncas), two_ct1_rdm(one_rdm, two_rdm, three_rdm)),
        - .5 * np.einsum('pu, nqlkrs->nulkpqrs', np.eye(ncas),
                         three_rdm),
        .5 * np.einsum('pu, lq, nkrs->nulkpqrs', *[np.eye(ncas)]*2,
                       two_rdm),
        - .5 * np.einsum('lu, nkpqrs->nulkpqrs', np.eye(ncas),
                         three_rdm)
    ))


def three_ct2_rdm(one_rdm, two_rdm, three_rdm, four_rdm):
    r"""
    .. math::
        \langle a_{m\sigma} E_{lk} E_{pq} E_{rs} a_{t\sigma}^\dagger \rangle

    Output index order: (tmlkpqrs)
    """
    ncas = one_rdm.shape[0]
    return sum((
        - .5 * four_rdm,
        np.einsum('mt, lkpqrs->tmlkpqrs', np.eye(ncas), three_rdm),
        np.einsum('st, rmlkpq->tmlkpqrs', np.eye(ncas), two_ct2_rdm(one_rdm, two_rdm, three_rdm)),
        np.einsum('qt, pm, lkrs->tmlkpqrs', np.eye(ncas), np.eye(ncas),
                  two_rdm),
        - .5 * np.einsum('qt, pmlkrs->tmlkpqrs', np.eye(ncas), three_rdm),
        np.einsum('qt, pk, lm, rs->tmlkpqrs', np.eye(ncas), np.eye(ncas),
                  np.eye(ncas), one_rdm),
        - .5 * np.einsum('qt, pk, lmrs->tmlkpqrs', np.eye(ncas),
                         np.eye(ncas), two_rdm),
        np.einsum('kt, lm, pqrs->tmlkpqrs', np.eye(ncas), np.eye(ncas),
                  two_rdm),
        - .5 * np.einsum('kt, lmpqrs->tmlkpqrs', np.eye(ncas), three_rdm)
    ))


def four_ct1_rdm(one_rdm, two_rdm, three_rdm, four_rdm, five_rdm):
    r"""
    .. math::
        \langle a_{n\sigma}^\dagger E_{lk} E_{pq} E_{rs} E_{tu} a_{w\sigma} \rangle
    """
    ncas = one_rdm.shape[0]
    return sum((
        .5 * five_rdm,
        - np.einsum('tw, nulkpqrs->nwlkpqrstu', np.eye(ncas),
                    three_ct1_rdm(one_rdm, two_rdm, three_rdm, four_rdm)),
        - .5 * np.einsum('rw, nslkpqtu->nwlkpqrstu', np.eye(ncas),
                         four_rdm),
        .5 * np.einsum('rw, ps, nqlktu->nwlkpqrstu', *[np.eye(ncas)]*2,
                       three_rdm),
        - .5 * np.einsum('rw, ps, lq, nktu->nwlkpqrstu', *[np.eye(ncas)]*3,
                         two_rdm),
        .5 * np.einsum('rw, ls, nkpqtu->nwlkpqrstu', *[np.eye(ncas)]*2,
                       three_rdm),
        - .5 * np.einsum('pw, nqlkrstu->nwlkpqrstu', np.eye(ncas),
                         four_rdm),
        .5 * np.einsum('pw, lq, nkrstu->nwlkpqrstu', *[np.eye(ncas)]*2,
                       three_rdm),
        - .5 * np.einsum('lw, nkpqrstu->nwlkpqrstu', np.eye(ncas),
                         four_rdm),
    ))


def four_ct2_rdm(one_rdm, two_rdm, three_rdm, four_rdm, five_rdm):
    r"""
    .. math::
        \langle a_{m\sigma} E_{lk} E_{pq} E_{rs} E_{tu} a_{v\sigma}^\dagger \rangle
    """
    ncas = one_rdm.shape[0]
    return sum((
        np.einsum('vm, lkpqrstu->vmlkpqrstu', np.eye(ncas),
                  four_rdm),
        - .5 * five_rdm,
        np.einsum('uv, tmlkpqrs->vmlkpqrstu', np.eye(ncas),
                  three_ct2_rdm(one_rdm, two_rdm, three_rdm, four_rdm)),
        np.einsum('sv, rm, lkpqtu->vmlkpqrstu', *[np.eye(ncas)]*2,
                  three_rdm),
        - .5 * np.einsum('sv, rmlkpqtu->vmlkpqrstu', np.eye(ncas),
                         four_rdm),
        np.einsum('sv, rq, pm, lktu->vmlkpqrstu', *[np.eye(ncas)]*3,
                  two_rdm),
        - .5 * np.einsum('sv, rq, pmlktu->vmlkpqrstu', np.eye(ncas),
                         np.eye(ncas), three_rdm),
        np.einsum('sv, rk, lm, pqtu->vmlkpqrstu', *[np.eye(ncas)]*3,
                  two_rdm),
        - .5 * np.einsum('sv, rk, lmpqtu->vmlkpqrstu', np.eye(ncas),
                         np.eye(ncas), three_rdm),
        np.einsum('sv, rq, pk, lm, tu->vmlkpqrstu', *[np.eye(ncas)]*4,
                  one_rdm),
        - .5 * np.einsum('sv, rq, pk, lmtu->vmlkpqrstu', *[np.eye(ncas)]*3,
                         two_rdm),
        np.einsum('qv, pm, lkrstu->vmlkpqrstu', *[np.eye(ncas)]*2,
                  three_rdm),
        - .5 * np.einsum('qv, pmlkrstu->vmlkpqrstu', np.eye(ncas),
                         four_rdm),
        np.einsum('qv, pk, lm, rstu->vmlkpqrstu', *[np.eye(ncas)]*3,
                  two_rdm),
        - .5 * np.einsum('qv, pk, lmrstu->vmlkpqrstu', *[np.eye(ncas)]*2,
                         three_rdm),
        np.einsum('kv, lm, pqrstu->vmlkpqrstu', *[np.eye(ncas)]*2,
                  three_rdm),
        - .5 * np.einsum('kv, lmpqrstu->vmlkpqrstu', np.eye(ncas),
                         four_rdm),
    ))
