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


# def get_tt_rdms(civec, ncas, nelec):
#     four_rdm_abba = make_dm4sabba(civec, ncas, nelec)
#     four_rdm_baab = make_dm4sbaab(civec, ncas, nelec)
#     four_rdm_aabb = make_dm4saabb(civec, ncas, nelec)

#     three_rdm_abba = make_dm3sabba(civec, ncas, nelec)
#     three_rdm_baab = make_dm3sbaab(civec, ncas, nelec)
#     three_rdm_aabb = make_dm3saabb(civec, ncas, nelec)

#     two_rdm_abba, two_rdm_baab, two_rdm_aabb = make_dm2s(
#         civec, ncas, nelec)

#     return (two_rdm_abba, two_rdm_baab, two_rdm_aabb,
#             three_rdm_abba, three_rdm_baab, three_rdm_aabb,
#             four_rdm_abba, four_rdm_baab, four_rdm_aabb)

def get_tt_rdms(civec, ncas, nelec):
    four_rdm_abba = make_dm4sabba(civec, ncas, nelec)
    four_rdm_baab = make_dm4sbaab(civec, ncas, nelec)
    four_rdm_aabb = make_dm4saabb(civec, ncas, nelec)

    three_rdm_abba = make_dm3sabba(civec, ncas, nelec)
    three_rdm_baab = make_dm3sbaab(civec, ncas, nelec)
    three_rdm_aabb = make_dm3saabb(civec, ncas, nelec)

    two_rdm_abba, two_rdm_baab, two_rdm_aabb = make_dm2s(
        civec, ncas, nelec)

    return [[two_rdm_abba, three_rdm_abba, four_rdm_abba],
            [two_rdm_baab, three_rdm_baab, four_rdm_baab],
            [two_rdm_aabb, three_rdm_aabb, four_rdm_aabb]]


def make_dm4sabba(civec, norb, nelec):
    r"""
    Construct the following spinful four-rdm:

    .. math::
        \langle \Psi | T^{(1,1)}_{lk} E_{pq} E_{rs}
        T^{(1,-1)}_{tu} | \Psi \rangle\\
        =-\langle \Psi | a^{\dagger}_{l \alpha} a_{k \beta} E_{pq}
        E_{rs} a^{\dagger}_{t \beta} a_{u \alpha} | \Psi \rangle

    """
    four_rdm_s14ab = np.zeros([norb]*8)
    nelec_a = nelec_b = nelec // 2

    for t, u, k, l in product(range(norb), repeat=4):
        civec_r_desa = des_a(civec, norb, (nelec_a, nelec_b), u)
        civec_r_credesab = cre_b(civec_r_desa, norb, (nelec_a-1, nelec_b), t)
        civec_l_desa = des_a(civec, norb, (nelec_a, nelec_b), l)
        civec_l_credesab = cre_b(civec_l_desa, norb, (nelec_a-1, nelec_b), k)
        (dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb) = trans_rdm12s(
            civec_l_credesab, civec_r_credesab, norb, (nelec_a-1, nelec_b+1),
            reorder=False)

        two_rdm_lktu = dm2aa + dm2bb + dm2ab + dm2ba
        four_rdm_s14ab[l, k, :, :, :, :, t, u] = -two_rdm_lktu
    return four_rdm_s14ab


def make_dm4sbaab(civec, norb, nelec):
    r"""
    Construct the following spinful four-rdm:

    .. math::
        \langle \Psi | T^{(1,-1)}_{lk} E_{pq} E_{rs}
        T^{(1,1)}_{tu} | \Psi \rangle\\
        =-\langle \Psi | a^{\dagger}_{l \beta} a_{k \alpha} E_{pq}
        E_{rs} a^{\dagger}_{t \alpha} a_{u \beta} | \Psi \rangle

    """
    four_rdm_s14ab = np.zeros([norb]*8)
    nelec_a = nelec_b = nelec // 2

    for t, u, k, l in product(range(norb), repeat=4):
        civec_r_desb = des_b(civec, norb, (nelec_a, nelec_b), u)
        civec_r_credesab = cre_a(civec_r_desb, norb, (nelec_a, nelec_b-1), t)
        civec_l_desb = des_b(civec, norb, (nelec_a, nelec_b), l)
        civec_l_credesab = cre_a(civec_l_desb, norb, (nelec_a, nelec_b-1), k)
        (dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb) = trans_rdm12s(
            civec_l_credesab, civec_r_credesab, norb, (nelec_a+1, nelec_b-1),
            reorder=False)

        two_rdm_lktu = dm2aa + dm2bb + dm2ab + dm2ba
        four_rdm_s14ab[l, k, :, :, :, :, t, u] = -two_rdm_lktu
    return four_rdm_s14ab


def make_dm4saabb(civec, norb, nelec):
    r"""
    Construct the following spinful four-rdm:

    .. math::
        \langle \Psi | T^{(1,0)}_{lk} E_{pq} E_{rs}
        T^{(1,0)}_{tu} | \Psi \rangle\\
        =\frac{1}{2}\bigg(\langle \Psi | a^{\dagger}_{l \alpha} a_{k \alpha} E_{pq}
        E_{rs} a^{\dagger}_{t \alpha} a_{u \alpha} | \Psi \rangle\\
        - \langle \Psi | a^{\dagger}_{l \alpha} a_{k \alpha} E_{pq}
        E_{rs} a^{\dagger}_{t \beta} a_{u \beta} | \Psi \rangle\\
        - \langle \Psi | a^{\dagger}_{l \beta} a_{k \beta} E_{pq}
        E_{rs} a^{\dagger}_{t \alpha} a_{u \alpha} | \Psi \rangle\\
        + \langle \Psi | a^{\dagger}_{l \beta} a_{k \beta} E_{pq}
        E_{rs} a^{\dagger}_{t \beta} a_{u \beta} | \Psi \rangle \bigg)

    """
    four_rdm_s14ab = np.zeros([norb]*8)
    nelec_a = nelec_b = nelec // 2

    for t, u, k, l in product(range(norb), repeat=4):
        civec_r_desa = des_a(civec, norb, (nelec_a, nelec_b), u)
        civec_r_desb = des_b(civec, norb, (nelec_a, nelec_b), u)

        civec_l_desa = des_a(civec, norb, (nelec_a, nelec_b), l)
        civec_l_desb = des_b(civec, norb, (nelec_a, nelec_b), l)

        civec_r_credesaa = cre_a(civec_r_desa, norb, (nelec_a-1, nelec_b), t)
        civec_r_credesbb = cre_b(civec_r_desb, norb, (nelec_a, nelec_b-1), t)

        civec_l_credesaa = cre_a(civec_l_desa, norb, (nelec_a-1, nelec_b), k)
        civec_l_credesbb = cre_b(civec_l_desb, norb, (nelec_a, nelec_b-1), k)

        (dm1a, dm1b), (dm2aa_aa, dm2ab_aa, dm2ba_aa, dm2bb_aa) = trans_rdm12s(
            civec_l_credesaa, civec_r_credesaa, norb, (nelec_a, nelec_b),
            reorder=False)

        (dm1a, dm1b), (dm2aa_ab, dm2ab_ab, dm2ba_ab, dm2bb_ab) = trans_rdm12s(
            civec_l_credesaa, civec_r_credesbb, norb, (nelec_a, nelec_b),
            reorder=False)

        (dm1a, dm1b), (dm2aa_ba, dm2ab_ba, dm2ba_ba, dm2bb_ba) = trans_rdm12s(
            civec_l_credesbb, civec_r_credesaa, norb, (nelec_a, nelec_b),
            reorder=False)

        (dm1a, dm1b), (dm2aa_bb, dm2ab_bb, dm2ba_bb, dm2bb_bb) = trans_rdm12s(
            civec_l_credesbb, civec_r_credesbb, norb, (nelec_a, nelec_b),
            reorder=False)

        two_rdm_lktu_aa = dm2aa_aa + dm2bb_aa + dm2ab_aa + dm2ba_aa

        two_rdm_lktu_ab = dm2aa_ab + dm2bb_ab + dm2ab_ab + dm2ba_ab

        two_rdm_lktu_ba = dm2aa_ba + dm2bb_ba + dm2ab_ba + dm2ba_ba

        two_rdm_lktu_bb = dm2aa_bb + dm2bb_bb + dm2ab_bb + dm2ba_bb

        four_rdm_s14ab[l, k, :, :, :, :, t, u] = .5 * sum((
            two_rdm_lktu_aa,
            - two_rdm_lktu_ab,
            - two_rdm_lktu_ba,
            two_rdm_lktu_bb
        ))
    return four_rdm_s14ab


def make_dm3sabba(civec, norb, nelec):
    r"""
    Construct the following spinful four-rdm:

    .. math::
        \langle \Psi | T^{(1,1)}_{lk} E_{pq}
        T^{(1,-1)}_{tu} | \Psi \rangle\\
        =-\langle \Psi | a^{\dagger}_{l \alpha} a_{k \beta} E_{pq}
        a^{\dagger}_{t \beta} a_{u \alpha} | \Psi \rangle

    """
    three_rdm_s14ab = np.zeros([norb]*6)
    nelec_a = nelec_b = nelec // 2

    for t, u, k, l in product(range(norb), repeat=4):
        civec_r_desa = des_a(civec, norb, (nelec_a, nelec_b), u)
        civec_r_credesab = cre_b(civec_r_desa, norb, (nelec_a-1, nelec_b), t)
        civec_l_desa = des_a(civec, norb, (nelec_a, nelec_b), l)
        civec_l_credesab = cre_b(civec_l_desa, norb, (nelec_a-1, nelec_b), k)
        dm1a, dm1b = trans_rdm1s(
            civec_l_credesab, civec_r_credesab, norb, (nelec_a-1, nelec_b+1))

        one_rdm_lktu = dm1a.T + dm1b.T
        three_rdm_s14ab[l, k, :, :, t, u] = -one_rdm_lktu
    return three_rdm_s14ab


def make_dm3sbaab(civec, norb, nelec):
    r"""
    Construct the following spinful four-rdm:

    .. math::
        \langle \Psi | T^{(1,-1)}_{lk} E_{pq}
        T^{(1,1)}_{tu} | \Psi \rangle\\
        =-\langle \Psi | a^{\dagger}_{l \beta} a_{k \alpha} E_{pq}
        a^{\dagger}_{t \alpha} a_{u \beta} | \Psi \rangle

    """
    three_rdm_s14ab = np.zeros([norb]*6)
    nelec_a = nelec_b = nelec // 2

    for t, u, k, l in product(range(norb), repeat=4):
        civec_r_desb = des_b(civec, norb, (nelec_a, nelec_b), u)
        civec_r_credesab = cre_a(civec_r_desb, norb, (nelec_a, nelec_b-1), t)
        civec_l_desb = des_b(civec, norb, (nelec_a, nelec_b), l)
        civec_l_credesab = cre_a(civec_l_desb, norb, (nelec_a, nelec_b-1), k)
        dm1a, dm1b = trans_rdm1s(
            civec_l_credesab, civec_r_credesab, norb, (nelec_a+1, nelec_b-1))

        one_rdm_lktu = dm1a.T + dm1b.T
        three_rdm_s14ab[l, k, :, :, t, u] = -one_rdm_lktu
    return three_rdm_s14ab


def make_dm3saabb(civec, norb, nelec):
    r"""
    Construct the following spinful four-rdm:

    .. math::
        \langle \Psi | T^{(1,0)}_{lk} E_{pq}
        T^{(1,0)}_{tu} | \Psi \rangle\\
        =\frac{1}{2}\bigg(\langle \Psi | a^{\dagger}_{l \alpha} a_{k \alpha} E_{pq}
        a^{\dagger}_{t \alpha} a_{u \alpha} | \Psi \rangle\\
        - \langle \Psi | a^{\dagger}_{l \alpha} a_{k \alpha} E_{pq}
        a^{\dagger}_{t \beta} a_{u \beta} | \Psi \rangle\\
        - \langle \Psi | a^{\dagger}_{l \beta} a_{k \beta} E_{pq}
        a^{\dagger}_{t \alpha} a_{u \alpha} | \Psi \rangle\\
        + \langle \Psi | a^{\dagger}_{l \beta} a_{k \beta} E_{pq}
        a^{\dagger}_{t \beta} a_{u \beta} | \Psi \rangle \bigg)

    """
    three_rdm_s14ab = np.zeros([norb]*6)
    nelec_a = nelec_b = nelec // 2

    for t, u, k, l in product(range(norb), repeat=4):
        civec_r_desa = des_a(civec, norb, (nelec_a, nelec_b), u)
        civec_r_desb = des_b(civec, norb, (nelec_a, nelec_b), u)

        civec_l_desa = des_a(civec, norb, (nelec_a, nelec_b), l)
        civec_l_desb = des_b(civec, norb, (nelec_a, nelec_b), l)

        civec_r_credesaa = cre_a(civec_r_desa, norb, (nelec_a-1, nelec_b), t)
        civec_r_credesbb = cre_b(civec_r_desb, norb, (nelec_a, nelec_b-1), t)

        civec_l_credesaa = cre_a(civec_l_desa, norb, (nelec_a-1, nelec_b), k)
        civec_l_credesbb = cre_b(civec_l_desb, norb, (nelec_a, nelec_b-1), k)

        dm1a_aa, dm1b_aa = trans_rdm1s(
            civec_l_credesaa, civec_r_credesaa, norb, (nelec_a, nelec_b))

        dm1a_ab, dm1b_ab = trans_rdm1s(
            civec_l_credesaa, civec_r_credesbb, norb, (nelec_a, nelec_b))

        dm1a_ba, dm1b_ba = trans_rdm1s(
            civec_l_credesbb, civec_r_credesaa, norb, (nelec_a, nelec_b))

        dm1a_bb, dm1b_bb = trans_rdm1s(
            civec_l_credesbb, civec_r_credesbb, norb, (nelec_a, nelec_b))

        one_rdm_lktu_aa = dm1a_aa.T + dm1b_aa.T

        one_rdm_lktu_ab = dm1a_ab.T + dm1b_ab.T

        one_rdm_lktu_ba = dm1a_ba.T + dm1b_ba.T

        one_rdm_lktu_bb = dm1a_bb.T + dm1b_bb.T

        three_rdm_s14ab[l, k, :, :, t, u] = .5 * sum((
            one_rdm_lktu_aa,
            - one_rdm_lktu_ab,
            - one_rdm_lktu_ba,
            one_rdm_lktu_bb
        ))
    return three_rdm_s14ab


def make_dm2s(civec, norb, nelec):
    r"""
    Construct the following spinful two-rdms:

    .. math::
        \langle \Psi | T^{(1,1)}_{lk} T^{(1,-1)}_{tu} | \Psi \rangle\\
        =-\langle \Psi | a^{\dagger}_{l \alpha} a_{k \beta}
        a^{\dagger}_{t \beta} a_{u \alpha} | \Psi \rangle

    and:

    .. math::
        \langle \Psi | T^{(1,-1)}_{lk}
        T^{(1,1)}_{tu} | \Psi \rangle\\
        =-\langle \Psi | a^{\dagger}_{l \beta} a_{k \alpha}
        a^{\dagger}_{t \alpha} a_{u \beta} | \Psi \rangle

    and:

    .. math::
        \langle \Psi | T^{(1,0)}_{lk} T^{(1,0)}_{tu} | \Psi \rangle\\
        =\frac{1}{2}\bigg(\langle \Psi | a^{\dagger}_{l \alpha} a_{k \alpha}
        a^{\dagger}_{t \alpha} a_{u \alpha} | \Psi \rangle\\
        - \langle \Psi | a^{\dagger}_{l \alpha} a_{k \alpha}
        a^{\dagger}_{t \beta} a_{u \beta} | \Psi \rangle\\
        - \langle \Psi | a^{\dagger}_{l \beta} a_{k \beta}
        a^{\dagger}_{t \alpha} a_{u \alpha} | \Psi \rangle\\
        + \langle \Psi | a^{\dagger}_{l \beta} a_{k \beta}
        a^{\dagger}_{t \beta} a_{u \beta} | \Psi \rangle \bigg)

    """
    (one_rdm_a, one_rdm_b), (two_rdm_aa, two_rdm_ab, two_rdm_bb) = make_rdm12s(
        civec, norb, nelec, reorder=False)

    two_rdm_abba = -two_rdm_ab.transpose(0, 3, 2, 1) + np.einsum(
        'tk,lu->lktu', np.eye(norb), one_rdm_a)

    two_rdm_baab = -two_rdm_ab.transpose(2, 1, 0, 3) + np.einsum(
        'tk,lu->lktu', np.eye(norb), one_rdm_b)

    two_rdm_aabb = two_rdm_aa - 2 * two_rdm_ab + two_rdm_bb

    return -two_rdm_abba, -two_rdm_baab, two_rdm_aabb
