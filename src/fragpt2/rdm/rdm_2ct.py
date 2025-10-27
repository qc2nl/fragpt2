#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 17:00:06 2024

@author: emielkoridon
"""


from itertools import product
import numpy as np

from pyscf.fci.direct_spin1 import trans_rdm12s
from pyscf.fci.addons import cre_a, cre_b, des_a, des_b


def overlap(bra, ket):
    return np.dot(bra.ravel().conj(), ket.ravel())


def get_2ct_rdms(ct, civec_l1, civec_l2, ncas_l1, nelec_l1, ncas_l2, nelec_l2):
    if ct == 'A':
        (two_2ct_rdm_abba_l1, three_2ct_rdm_abba_l1,
         four_2ct_rdm_abba_l1) = make_dm234_2ct_cre_abba(civec_l1, ncas_l1, nelec_l1)
        (two_2ct_rdm_aaaa_l1, three_2ct_rdm_aaaa_l1,
         four_2ct_rdm_aaaa_l1) = make_dm234_2ct_cre_aaaa(civec_l1, ncas_l1, nelec_l1)
        (two_2ct_rdm_bbbb_l1, three_2ct_rdm_bbbb_l1,
         four_2ct_rdm_bbbb_l1) = make_dm234_2ct_cre_bbbb(civec_l1, ncas_l1, nelec_l1)

        (two_2ct_rdm_abba_l2, three_2ct_rdm_abba_l2,
         four_2ct_rdm_abba_l2) = make_dm234_2ct_des_abba(civec_l2, ncas_l2, nelec_l2)
        (two_2ct_rdm_aaaa_l2, three_2ct_rdm_aaaa_l2,
         four_2ct_rdm_aaaa_l2) = make_dm234_2ct_des_aaaa(civec_l2, ncas_l2, nelec_l2)
        (two_2ct_rdm_bbbb_l2, three_2ct_rdm_bbbb_l2,
         four_2ct_rdm_bbbb_l2) = make_dm234_2ct_des_bbbb(civec_l2, ncas_l2, nelec_l2)

    elif ct == 'B':
        (two_2ct_rdm_abba_l1, three_2ct_rdm_abba_l1,
         four_2ct_rdm_abba_l1) = make_dm234_2ct_des_abba(civec_l1, ncas_l1, nelec_l1)
        (two_2ct_rdm_aaaa_l1, three_2ct_rdm_aaaa_l1,
         four_2ct_rdm_aaaa_l1) = make_dm234_2ct_des_aaaa(civec_l1, ncas_l1, nelec_l1)
        (two_2ct_rdm_bbbb_l1, three_2ct_rdm_bbbb_l1,
         four_2ct_rdm_bbbb_l1) = make_dm234_2ct_des_bbbb(civec_l1, ncas_l1, nelec_l1)

        (two_2ct_rdm_abba_l2, three_2ct_rdm_abba_l2,
         four_2ct_rdm_abba_l2) = make_dm234_2ct_cre_abba(civec_l2, ncas_l2, nelec_l2)
        (two_2ct_rdm_aaaa_l2, three_2ct_rdm_aaaa_l2,
         four_2ct_rdm_aaaa_l2) = make_dm234_2ct_cre_aaaa(civec_l2, ncas_l2, nelec_l2)
        (two_2ct_rdm_bbbb_l2, three_2ct_rdm_bbbb_l2,
         four_2ct_rdm_bbbb_l2) = make_dm234_2ct_cre_bbbb(civec_l2, ncas_l2, nelec_l2)

    return ([two_2ct_rdm_abba_l1, three_2ct_rdm_abba_l1, four_2ct_rdm_abba_l1],
            [two_2ct_rdm_aaaa_l1, three_2ct_rdm_aaaa_l1, four_2ct_rdm_aaaa_l1],
            [two_2ct_rdm_bbbb_l1, three_2ct_rdm_bbbb_l1, four_2ct_rdm_bbbb_l1],
            [two_2ct_rdm_abba_l2, three_2ct_rdm_abba_l2, four_2ct_rdm_abba_l2],
            [two_2ct_rdm_aaaa_l2, three_2ct_rdm_aaaa_l2, four_2ct_rdm_aaaa_l2],
            [two_2ct_rdm_bbbb_l2, three_2ct_rdm_bbbb_l2, four_2ct_rdm_bbbb_l2])


def make_dm234_2ct_des_abba(civec, norb, nelec):
    r"""
    Construct the following spinful four-rdm:

    .. math::
        \langle \Psi | a^{\dagger}_{m \alpha} a^{\dagger}_{k \beta} E_{pq}
        E_{rs} a_{t \beta} a_{v \alpha} | \Psi \rangle

    """
    four_rdm_s14ab = np.zeros([norb]*8)
    three_rdm_s14ab = np.zeros([norb]*6)
    two_rdm_s14ab = np.zeros([norb]*4)
    nelec_a = nelec_b = nelec // 2
    for t, v, k, m in product(range(norb), repeat=4):
        civec_r_desa = des_a(civec, norb, (nelec_a, nelec_b), v)
        civec_r_desdesab = des_b(civec_r_desa, norb, (nelec_a-1, nelec_b), t)
        civec_l_desa = des_a(civec, norb, (nelec_a, nelec_b), m)
        civec_l_desdesab = des_b(civec_l_desa, norb, (nelec_a-1, nelec_b), k)
        (dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb) = trans_rdm12s(
            civec_l_desdesab, civec_r_desdesab, norb, (nelec_a-1, nelec_b-1),
            reorder=False)

        two_rdm_lktu = dm2aa + dm2bb + dm2ab + dm2ba
        one_rdm_lktu = dm1a + dm1b
        zero_rdm_lktu = overlap(civec_l_desdesab, civec_r_desdesab)
        four_rdm_s14ab[m, k, :, :, :, :, t, v] = two_rdm_lktu
        three_rdm_s14ab[m, k, :, :, t, v] = one_rdm_lktu
        two_rdm_s14ab[m, k, t, v] = zero_rdm_lktu
    return two_rdm_s14ab, three_rdm_s14ab, four_rdm_s14ab


def make_dm234_2ct_des_aaaa(civec, norb, nelec):
    r"""
    Construct the following spinful four-rdm:

    .. math::
        \langle \Psi | a^{\dagger}_{m \alpha} a^{\dagger}_{k \alpha} E_{pq}
        E_{rs} a_{t \alpha} a_{v \alpha} | \Psi \rangle

    """
    four_rdm_s14ab = np.zeros([norb]*8)
    three_rdm_s14ab = np.zeros([norb]*6)
    two_rdm_s14ab = np.zeros([norb]*4)
    nelec_a = nelec_b = nelec // 2
    if not nelec_a - 2 < 0:
        for t, v, k, m in product(range(norb), repeat=4):
            civec_r_desa = des_a(civec, norb, (nelec_a, nelec_b), v)
            civec_r_desdesaa = des_a(civec_r_desa, norb, (nelec_a-1, nelec_b), t)
            civec_l_desa = des_a(civec, norb, (nelec_a, nelec_b), m)
            civec_l_desdesaa = des_a(civec_l_desa, norb, (nelec_a-1, nelec_b), k)
            (dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb) = trans_rdm12s(
                civec_l_desdesaa, civec_r_desdesaa, norb, (nelec_a-2, nelec_b),
                reorder=False)

            two_rdm_lktu = dm2aa + dm2bb + dm2ab + dm2ba
            one_rdm_lktu = dm1a + dm1b
            zero_rdm_lktu = overlap(civec_l_desdesaa, civec_r_desdesaa)
            four_rdm_s14ab[m, k, :, :, :, :, t, v] = two_rdm_lktu
            three_rdm_s14ab[m, k, :, :, t, v] = one_rdm_lktu
            two_rdm_s14ab[m, k, t, v] = zero_rdm_lktu
    return two_rdm_s14ab, three_rdm_s14ab, four_rdm_s14ab


def make_dm234_2ct_des_bbbb(civec, norb, nelec):
    r"""
    Construct the following spinful four-rdm:

    .. math::
        \langle \Psi | a^{\dagger}_{m \beta} a^{\dagger}_{k \beta} E_{pq}
        E_{rs} a_{t \beta} a_{v \beta} | \Psi \rangle

    """
    four_rdm_s14ab = np.zeros([norb]*8)
    three_rdm_s14ab = np.zeros([norb]*6)
    two_rdm_s14ab = np.zeros([norb]*4)
    nelec_a = nelec_b = nelec // 2
    if not nelec_b - 2 < 0:
        for t, v, k, m in product(range(norb), repeat=4):
            civec_r_desb = des_b(civec, norb, (nelec_a, nelec_b), v)
            civec_r_desdesbb = des_b(civec_r_desb, norb, (nelec_a, nelec_b-1), t)
            civec_l_desb = des_b(civec, norb, (nelec_a, nelec_b), m)
            civec_l_desdesbb = des_b(civec_l_desb, norb, (nelec_a, nelec_b-1), k)
            (dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb) = trans_rdm12s(
                civec_l_desdesbb, civec_r_desdesbb, norb, (nelec_a, nelec_b-2),
                reorder=False)

            two_rdm_lktu = dm2aa + dm2bb + dm2ab + dm2ba
            one_rdm_lktu = dm1a + dm1b
            zero_rdm_lktu = overlap(civec_l_desdesbb, civec_r_desdesbb)
            four_rdm_s14ab[m, k, :, :, :, :, t, v] = two_rdm_lktu
            three_rdm_s14ab[m, k, :, :, t, v] = one_rdm_lktu
            two_rdm_s14ab[m, k, t, v] = zero_rdm_lktu
    return two_rdm_s14ab, three_rdm_s14ab, four_rdm_s14ab


def make_dm234_2ct_cre_abba(civec, norb, nelec):
    r"""
    Construct the following spinful four-rdm:

    .. math::
        \langle \Psi | a_{m \alpha} a_{k \beta} E_{pq}
        E_{rs} a^{\dagger}_{t \beta} a^\dagger_{v \alpha} | \Psi \rangle

    """
    four_rdm_s14ab = np.zeros([norb]*8)
    three_rdm_s14ab = np.zeros([norb]*6)
    two_rdm_s14ab = np.zeros([norb]*4)
    nelec_a = nelec_b = nelec // 2

    for t, v, k, m in product(range(norb), repeat=4):
        civec_r_crea = cre_a(civec, norb, (nelec_a, nelec_b), v)
        civec_r_crecreab = cre_b(civec_r_crea, norb, (nelec_a+1, nelec_b), t)
        civec_l_crea = cre_a(civec, norb, (nelec_a, nelec_b), m)
        civec_l_crecreab = cre_b(civec_l_crea, norb, (nelec_a+1, nelec_b), k)
        (dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb) = trans_rdm12s(
            civec_l_crecreab, civec_r_crecreab, norb, (nelec_a+1, nelec_b+1),
            reorder=False)

        two_rdm_lktu = dm2aa + dm2bb + dm2ab + dm2ba
        one_rdm_lktu = dm1a + dm1b
        zero_rdm_lktu = overlap(civec_l_crecreab, civec_r_crecreab)
        four_rdm_s14ab[m, k, :, :, :, :, t, v] = two_rdm_lktu
        three_rdm_s14ab[m, k, :, :, t, v] = one_rdm_lktu
        two_rdm_s14ab[m, k, t, v] = zero_rdm_lktu
    return two_rdm_s14ab, three_rdm_s14ab, four_rdm_s14ab


def make_dm234_2ct_cre_aaaa(civec, norb, nelec):
    r"""
    Construct the following spinful four-rdm:

    .. math::
        \langle \Psi | a_{m \alpha} a_{k \alpha} E_{pq}
        E_{rs} a^{\dagger}_{t \alpha} a^\dagger_{v \alpha} | \Psi \rangle

    """
    four_rdm_s14ab = np.zeros([norb]*8)
    three_rdm_s14ab = np.zeros([norb]*6)
    two_rdm_s14ab = np.zeros([norb]*4)
    nelec_a = nelec_b = nelec // 2
    if not nelec_a + 2 > norb:
        for t, v, k, m in product(range(norb), repeat=4):
            civec_r_crea = cre_a(civec, norb, (nelec_a, nelec_b), v)
            civec_r_crecreaa = cre_a(civec_r_crea, norb, (nelec_a+1, nelec_b), t)
            civec_l_crea = cre_a(civec, norb, (nelec_a, nelec_b), m)
            civec_l_crecreaa = cre_a(civec_l_crea, norb, (nelec_a+1, nelec_b), k)
            (dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb) = trans_rdm12s(
                civec_l_crecreaa, civec_r_crecreaa, norb, (nelec_a+2, nelec_b),
                reorder=False)

            two_rdm_lktu = dm2aa + dm2bb + dm2ab + dm2ba
            one_rdm_lktu = dm1a + dm1b
            zero_rdm_lktu = overlap(civec_l_crecreaa, civec_r_crecreaa)
            four_rdm_s14ab[m, k, :, :, :, :, t, v] = two_rdm_lktu
            three_rdm_s14ab[m, k, :, :, t, v] = one_rdm_lktu
            two_rdm_s14ab[m, k, t, v] = zero_rdm_lktu
    return two_rdm_s14ab, three_rdm_s14ab, four_rdm_s14ab


def make_dm234_2ct_cre_bbbb(civec, norb, nelec):
    r"""
    Construct the following spinful four-rdm:

    .. math::
        \langle \Psi | a_{m \beta} a_{k \beta} E_{pq}
        E_{rs} a^{\dagger}_{t \beta} a^\dagger_{v \beta} | \Psi \rangle

    """
    four_rdm_s14ab = np.zeros([norb]*8)
    three_rdm_s14ab = np.zeros([norb]*6)
    two_rdm_s14ab = np.zeros([norb]*4)
    nelec_a = nelec_b = nelec // 2
    if not nelec_b + 2 > norb:
        for t, v, k, m in product(range(norb), repeat=4):
            civec_r_creb = cre_b(civec, norb, (nelec_a, nelec_b), v)
            civec_r_crecrebb = cre_b(civec_r_creb, norb, (nelec_a, nelec_b+1), t)
            civec_l_creb = cre_b(civec, norb, (nelec_a, nelec_b), m)
            civec_l_crecrebb = cre_b(civec_l_creb, norb, (nelec_a, nelec_b+1), k)
            (dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb) = trans_rdm12s(
                civec_l_crecrebb, civec_r_crecrebb, norb, (nelec_a, nelec_b+2),
                reorder=False)

            two_rdm_lktu = dm2aa + dm2bb + dm2ab + dm2ba
            one_rdm_lktu = dm1a + dm1b
            zero_rdm_lktu = overlap(civec_l_crecrebb, civec_r_crecrebb)
            four_rdm_s14ab[m, k, :, :, :, :, t, v] = two_rdm_lktu
            three_rdm_s14ab[m, k, :, :, t, v] = one_rdm_lktu
            two_rdm_s14ab[m, k, t, v] = zero_rdm_lktu
    return two_rdm_s14ab, three_rdm_s14ab, four_rdm_s14ab
