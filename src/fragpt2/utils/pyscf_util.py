#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 12:02:14 2023

@author: emielkoridon
"""
import re
import numpy as np
import os
import pyscf
from pyscf.tools import cubegen
from pyscf.fci.direct_spin0 import make_rdm1, make_rdm12
import ast


def read_int(text, f):
    for line in f:
        if re.search(text, line):
            var = int(line.rsplit(None, 1)[-1])
            return var


def read_real_list(text, f):
    for line in f:
        if re.search(text, line):
            n = int(line.rsplit(None, 1)[-1])
            var = []
            for i in range((n-1)//5+1):
                line = next(f)
                for j in line.split():
                    var += [float(j)]
            return var


def read_MO_list(fname):
    mo_list = {}
    with open(fname, "r") as f:
        for line in f:
            try:
                mo_list[int(line.split()[0])] = [ast.literal_eval(j) for j in line.split()[1:]]
            except:
                mo_list[float(line.split()[0])] = [ast.literal_eval(j) for j in line.split()[1:]]

    return mo_list


def unpack_pyscf(fname):
    with open(fname, "r") as f:
        nao = read_int("Number of basis functions", f)
        alpha_energies = read_real_list("Alpha Orbital Energies", f)
        alpha_IBO = read_real_list("Alpha MO coefficients", f)
        energy = read_real_list("scf_e", f)
    f.close()
    nmo = len(alpha_energies)
    alpha_IBO_coeff = np.array(alpha_IBO).reshape(nmo, nao).T
    return nao, np.array(alpha_energies), alpha_IBO_coeff


def make_cube(mo_coeff, mol, datadir, description, indices):
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    for i in indices:
        cubegen.orbital(mol, os.path.join(datadir, description +
                        '_orb_' + str(i) + '.cube'), mo_coeff[:, i])


def get_expectation(one_body_integrals, two_body_integrals, civec, norb, nelec):
    """
    Compute expectation of two-body observable defined by its one- and two-body
    spatial integrals given CI coefficients.

    Args:
        one_body_integrals (np.ndarray):
            One-body integrals.
        two_body_integrals (np.ndarray):
            Two-body integrals (factor .5 assumed, molecular integral convention)
        civec (pyscf.fci.FCIvector):
            PySCF FCIvector defining the CI state
        norb (int):
            Number of (active) orbitals
        nelec (int):
            Number of (active) electrons

    Returns:
        int: Expectation value.

    """
    one_rdm, two_rdm = make_rdm12(civec, norb, nelec, reorder=True)
    return (
        np.einsum('pq, qp', one_body_integrals, one_rdm),
        np.einsum('pqrs, pqrs', two_body_integrals, two_rdm)
    )


def pyscf_ci_to_psi(civec, ncas, nelecas):
    psi = np.zeros(2**(2*ncas))
    occslst = pyscf.fci.cistring.gen_occslst(range(ncas), nelecas//2)
    for i, occsa in enumerate(occslst):
        for j, occsb in enumerate(occslst):
            alpha_bin = [1 if x in occsa else 0 for x in range(ncas)]
            beta_bin = [1 if y in occsb else 0 for y in range(ncas)]
            alpha_bin.reverse()
            beta_bin.reverse()
            idx = 0
            for spatorb in range(ncas):
                if alpha_bin[spatorb] == 1:
                    idx += 2**(spatorb + ncas)
                if beta_bin[spatorb] == 1:
                    idx += 2**(spatorb)
            psi[idx] = civec[i, j]
    return psi
