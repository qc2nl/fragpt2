#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 15:47:16 2023

@author: emielkoridon
"""
import numpy as np


def active_space_integrals(one_body_integrals,
                           two_body_integrals,
                           occ_idx,
                           act_idx):
    """
    Restricts a molecule at a spatial orbital level to an active space
    This active space may be defined by a list of active indices and
    doubly occupied indices. Note that one_body_integrals and
    two_body_integrals must be defined in an orthonormal basis set (MO like).

    NB: Chemists's ordering convention required for the two-body integrals

    ---

    Args:
         - one_body_integrals: spatial [p^ q] integrals
         - two_body_integrals: spatial [p^ r^ s q] integrals (chemist's order)
         - occ_idx: A list of spatial orbital indices
           indicating which orbitals should be considered doubly occupied.
         - act_idx: A list of spatial orbital indices indicating
           which orbitals should be considered active.

    Returns:
        `tuple`: Tuple with the following entries:
            - core_constant: Adjustment to constant shift in Hamiltonian
                from integrating out core orbitals
            - as_one_body_integrals: one-electron integrals over active space.
            - as_two_body_integrals: two-electron integrals over active space.
    """

    obai = np.ix_(*[act_idx]*2)
    tbai = np.ix_(*[act_idx]*4)
    # --- Determine core constant ---
    core_constant = (
        2 * np.sum(one_body_integrals[occ_idx, occ_idx]) +  # i^ j
        2 * np.sum(two_body_integrals[
            occ_idx, occ_idx, :, :][
            :, occ_idx, occ_idx])  # i^ j^ j i
        - np.sum(two_body_integrals[
            occ_idx, :, :, occ_idx][
            :, occ_idx, occ_idx])  # i^ j^ i j
    )

    # restrict range to active indices only
    as_two_body_integrals = two_body_integrals[tbai]

    # --- Modified one electron integrals ---
    # sum over i in occ_idx
    as_one_body_integrals = (
        one_body_integrals[obai]
        + 2 * np.sum(two_body_integrals[:, :, occ_idx, occ_idx
                                        ][act_idx, :, :][:, act_idx, :],
                     axis=2)  # i^ p^ q i
        - np.sum(two_body_integrals[:, occ_idx, occ_idx, :][
            act_idx, :, :][:, :, act_idx], axis=1)  # i^ p^ i q
    )

    # Restrict integral ranges and change M appropriately
    return (core_constant,
            as_one_body_integrals,
            as_two_body_integrals)


def molecular_hamiltonian_coefficients(nuclear_repulsion,
                                       one_body_integrals,
                                       two_body_integrals,
                                       occ_idx=None,
                                       act_idx=None):
    '''
    Transform full-space restricted orbitals to CAS restricted Hamiltonian
    coefficient's in chemist notation.

    The resulting tensors are ready for openfermion.InteractionOperator, and
    follow the same conventions

    Returns: `tuple` consisting of
        - E_constant (one dimensional tensor)
        - one_body_coefficients (2-dimensional tensor)
        - two_body_coefficients (4-dimensional tensor)

    '''

    # Build CAS
    if occ_idx is None and act_idx is None:
        E_constant = nuclear_repulsion
    else:
        (core_adjustment,
         one_body_integrals,
         two_body_integrals) = active_space_integrals(one_body_integrals,
                                                      two_body_integrals,
                                                      occ_idx,
                                                      act_idx)
        E_constant = core_adjustment + nuclear_repulsion

    # Initialize Hamiltonian coefficients.
    one_body_coefficients = one_body_integrals
    two_body_coefficients = two_body_integrals

    return E_constant, one_body_coefficients, two_body_coefficients


def get_active_integrals(one_body_integrals, two_body_integrals, occ_idx, act_idx,
                         nuclear_repulsion=None, with_nuc=False):
    """Get active space integrals (zero, one and two-body). See documentation of
    `molecular_hamiltonian_coefficients`."""
    if with_nuc:
        assert nuclear_repulsion is not None
        return molecular_hamiltonian_coefficients(
            nuclear_repulsion, one_body_integrals, two_body_integrals,
            occ_idx, act_idx)
    else:
        return molecular_hamiltonian_coefficients(
            0, one_body_integrals, two_body_integrals,
            occ_idx, act_idx)
