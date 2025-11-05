#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 17:57:59 2023

@author: emielkoridon
"""

import numpy as np

from pyscf import gto, scf
from fragpt2 import unpack_pyscf, FragPT2


basis = 'unc-cc-pvdz'
symmetry = 0
cart = True

mol = gto.Mole(atom='water-ammonia.xyz',
               basis=basis, symmetry=symmetry, cart=cart)
mol.max_memory = 8000

mol.build()
mf = scf.RHF(mol)
mf.run()

nao, mo_energies, mo_coeff = unpack_pyscf("ribo.pyscf")

# # # # for CAS(4,4)(4,4):
act_idx_l1_base1 = [4, 5, 11, 12]
act_idx_l2_base1 = [8, 9, 27, 28]


act_idx_l1 = np.array(act_idx_l1_base1) - 1
act_idx_l2 = np.array(act_idx_l2_base1) - 1
act_idx = np.concatenate((act_idx_l1, act_idx_l2))

# SET HERE THE SIZE OF THE ACTIVE SPACE
ncas = len(act_idx)
nelecas = ncas - 2 * (ncas // 2 % 2)
nelec_l1 = nelecas // 2
nelec_l2 = nelecas // 2
ncas_l1 = ncas // 2
ncas_l2 = ncas // 2


print(f'l1: CAS({nelec_l1},{ncas_l1})')
print(f'l2: CAS({nelec_l2},{ncas_l2})')

occ_idx = [x for x in range(mol.nelectron//2) if x not in act_idx]

nroots = 1

casci = FragPT2(mol, occ_idx, act_idx,
                ((ncas_l1, nelec_l1), (ncas_l2, nelec_l2)),
                mo_energies, mo_coeff)

casci.run_mean_field()
casci.run_full_casci(nroots)
casci.run_l1_casci(nroots)
casci.run_l2_casci(nroots)
casci.run_ct1_casci(nroots)
casci.run_ct2_casci(nroots)
e_tot, e_l1, e_l2 = casci.run_self_consistent_casci()

e_corr_naive = casci.e_l1 + casci.e_l2 - 2 * mf.e_tot
e_sumcorr = casci.e_l1 + casci.e_l2 - mf.e_tot

print(f'Naive energy:            {casci.e_naive:.6f}')
print(f'Sum of corr energies:    {e_sumcorr:.6f}')
print(f'scf energy:              {e_tot:.6f}')
print(f'exact energy:            {casci.e_full:.6f}')

e_pt2_disp = casci.run_disp()

print(f'Pt2 disp correction:     {e_pt2_disp:.8f}')
print(f'embedding energy wo pt3: {e_tot + e_pt2_disp:.6f}')

e_ct1_ci = casci.e_ct1 - mf.e_tot
e_ct2_ci = casci.e_ct2 - mf.e_tot

print(f'CT1 CI energy:        {e_ct1_ci:.6f}')
print(f'CT2 CI energy:        {e_ct2_ci:.6f}')

e_pt2_ct1_1, e_pt2_ct1_2 = casci.run_1ct()

print(f'CT1_1 PT2 energy:     {e_pt2_ct1_1:.6f}')
print(f'CT1_2 PT2 energy:     {e_pt2_ct1_2:.6f}')

e_pt2_tt = casci.run_tt()

print(f'PT2 TT correction:    {e_pt2_tt:.12f}')

e_pt2_ct2_1, e_pt2_ct2_2 = casci.run_2ct()

print(f'CT2_1 PT2 energy:     {e_pt2_ct2_1:.12f}')
print(f'CT2_2 PT2 energy:     {e_pt2_ct2_2:.12f}')

e_pt2_ct1 = e_pt2_ct1_1 + e_pt2_ct1_2
e_pt2_ct2 = e_pt2_ct2_1 + e_pt2_ct2_2

print(
    f'Full embedding E:     {e_tot + e_pt2_disp + e_pt2_tt + (e_pt2_ct1 + e_pt2_ct2):.6f}')
