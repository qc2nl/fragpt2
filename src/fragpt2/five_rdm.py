#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:06:44 2023

@author: emielkoridon
"""

import numpy as np
import time

from pyscf import gto, scf
from fragpt2.fragpt2 import FragPT2, h_prime
from fragpt2.rdm_utils import (
    compute_five_rdm, make_dm2s, make_dm3saabb, make_dm3sabba, make_dm3sbaab,
    make_dm4saabb, make_dm4sabba, make_dm4sbaab)


class Exact_1ct(FragPT2):
    def __init__(self, mf: scf.hf.RHF,
                 occ_idx, act_idx,
                 fragments,
                 mo_energies=None, mo_coeff=None, verbose=0):

        super().__init__(mf, occ_idx, act_idx, fragments,
                         mo_energies=mo_energies, mo_coeff=mo_coeff, verbose=verbose)
        self.five_rdm_l1 = None
        self.five_rdm_l2 = None

        self.got_tt_rdms = False

    def get_five_rdms(self, again=False):
        if self.five_rdm_l1 is None or self.five_rdm_l2 is None or again:
            t1 = time.time()
            self.five_rdm_l1 = compute_five_rdm(
                self.civec_l1, self.ncas_l1, self.nelec_l1)
            self.five_rdm_l2 = compute_five_rdm(
                self.civec_l2, self.ncas_l2, self.nelec_l2)
            print('computing 5-RDMs took:', time.time()-t1)

    def three_ct1_rdm(self, fragment):
        r"""
        .. math::
            \langle a_{n\sigma}^\dagger E_{lk} E_{pq} E_{rs} a_{u\sigma} \rangle
        """
        if fragment == 'l1':
            return sum((
                .5 * self.four_rdm_l1,
                - np.einsum('ru, nslkpq->nulkpqrs',
                            np.eye(self.ncas_l1), self.two_ct1_rdm('l1')),
                - .5 * np.einsum('pu, nqlkrs->nulkpqrs', np.eye(self.ncas_l1),
                                 self.three_rdm_l1),
                .5 * np.einsum('pu, lq, nkrs->nulkpqrs', *[np.eye(self.ncas_l1)]*2,
                               self.two_rdm_l1_ord),
                - .5 * np.einsum('lu, nkpqrs->nulkpqrs', np.eye(self.ncas_l1),
                                 self.three_rdm_l1)
            ))
        elif fragment == 'l2':
            return sum((
                .5 * self.four_rdm_l2,
                - np.einsum('ru, nslkpq->nulkpqrs',
                            np.eye(self.ncas_l2), self.two_ct1_rdm('l2')),
                - .5 * np.einsum('pu, nqlkrs->nulkpqrs', np.eye(self.ncas_l2),
                                 self.three_rdm_l2),
                .5 * np.einsum('pu, lq, nkrs->nulkpqrs', *[np.eye(self.ncas_l2)]*2,
                               self.two_rdm_l2_ord),
                - .5 * np.einsum('lu, nkpqrs->nulkpqrs', np.eye(self.ncas_l2),
                                 self.three_rdm_l2)
            ))

    def three_ct2_rdm(self, fragment):
        r"""
        .. math::
            \langle a_{m\sigma} E_{lk} E_{pq} E_{rs} a_{t\sigma}^\dagger \rangle

        Output index order: (tmlkpqrs)
        """
        if fragment == 'l1':
            return sum((
                - .5 * self.four_rdm_l1,
                np.einsum('mt, lkpqrs->tmlkpqrs',
                          np.eye(self.ncas_l1), self.three_rdm_l1),
                np.einsum('st, rmlkpq->tmlkpqrs',
                          np.eye(self.ncas_l1), self.two_ct2_rdm('l1')),
                np.einsum('qt, pm, lkrs->tmlkpqrs', np.eye(self.ncas_l1), np.eye(self.ncas_l1),
                          self.two_rdm_l1_ord),
                - .5 * np.einsum('qt, pmlkrs->tmlkpqrs',
                                 np.eye(self.ncas_l1), self.three_rdm_l1),
                np.einsum('qt, pk, lm, rs->tmlkpqrs', np.eye(self.ncas_l1), np.eye(self.ncas_l1),
                          np.eye(self.ncas_l1), self.one_rdm_l1),
                - .5 * np.einsum('qt, pk, lmrs->tmlkpqrs', np.eye(self.ncas_l1),
                                 np.eye(self.ncas_l1), self.two_rdm_l1_ord),
                np.einsum('kt, lm, pqrs->tmlkpqrs', np.eye(self.ncas_l1), np.eye(self.ncas_l1),
                          self.two_rdm_l1_ord),
                - .5 * np.einsum('kt, lmpqrs->tmlkpqrs',
                                 np.eye(self.ncas_l1), self.three_rdm_l1)
            ))
        elif fragment == 'l2':
            return sum((
                - .5 * self.four_rdm_l2,
                np.einsum('mt, lkpqrs->tmlkpqrs',
                          np.eye(self.ncas_l2), self.three_rdm_l2),
                np.einsum('st, rmlkpq->tmlkpqrs',
                          np.eye(self.ncas_l2), self.two_ct2_rdm('l2')),
                np.einsum('qt, pm, lkrs->tmlkpqrs', np.eye(self.ncas_l2), np.eye(self.ncas_l2),
                          self.two_rdm_l2_ord),
                - .5 * np.einsum('qt, pmlkrs->tmlkpqrs',
                                 np.eye(self.ncas_l2), self.three_rdm_l2),
                np.einsum('qt, pk, lm, rs->tmlkpqrs', np.eye(self.ncas_l2), np.eye(self.ncas_l2),
                          np.eye(self.ncas_l2), self.one_rdm_l2),
                - .5 * np.einsum('qt, pk, lmrs->tmlkpqrs', np.eye(self.ncas_l2),
                                 np.eye(self.ncas_l2), self.two_rdm_l2_ord),
                np.einsum('kt, lm, pqrs->tmlkpqrs', np.eye(self.ncas_l2), np.eye(self.ncas_l2),
                          self.two_rdm_l2_ord),
                - .5 * np.einsum('kt, lmpqrs->tmlkpqrs',
                                 np.eye(self.ncas_l2), self.three_rdm_l2)
            ))

    def four_ct1_rdm(self, fragment):
        r"""
        .. math::
            \langle a_{n\sigma}^\dagger E_{lk} E_{pq} E_{rs} E_{tu} a_{w\sigma} \rangle
        """
        if fragment == 'l1':
            return sum((
                .5 * self.five_rdm_l1,
                - np.einsum('tw, nulkpqrs->nwlkpqrstu', np.eye(self.ncas_l1),
                            self.three_ct1_rdm('l1')),
                - .5 * np.einsum('rw, nslkpqtu->nwlkpqrstu', np.eye(self.ncas_l1),
                                 self.four_rdm_l1),
                .5 * np.einsum('rw, ps, nqlktu->nwlkpqrstu', *[np.eye(self.ncas_l1)]*2,
                               self.three_rdm_l1),
                - .5 * np.einsum('rw, ps, lq, nktu->nwlkpqrstu', *[np.eye(self.ncas_l1)]*3,
                                 self.two_rdm_l1_ord),
                .5 * np.einsum('rw, ls, nkpqtu->nwlkpqrstu', *[np.eye(self.ncas_l1)]*2,
                               self.three_rdm_l1),
                - .5 * np.einsum('pw, nqlkrstu->nwlkpqrstu', np.eye(self.ncas_l1),
                                 self.four_rdm_l1),
                .5 * np.einsum('pw, lq, nkrstu->nwlkpqrstu', *[np.eye(self.ncas_l1)]*2,
                               self.three_rdm_l1),
                - .5 * np.einsum('lw, nkpqrstu->nwlkpqrstu', np.eye(self.ncas_l1),
                                 self.four_rdm_l1),
            ))
        elif fragment == 'l2':
            return sum((
                .5 * self.five_rdm_l2,
                - np.einsum('tw, nulkpqrs->nwlkpqrstu', np.eye(self.ncas_l2),
                            self.three_ct1_rdm('l2')),
                - .5 * np.einsum('rw, nslkpqtu->nwlkpqrstu', np.eye(self.ncas_l2),
                                 self.four_rdm_l2),
                .5 * np.einsum('rw, ps, nqlktu->nwlkpqrstu', *[np.eye(self.ncas_l2)]*2,
                               self.three_rdm_l2),
                - .5 * np.einsum('rw, ps, lq, nktu->nwlkpqrstu', *[np.eye(self.ncas_l2)]*3,
                                 self.two_rdm_l2_ord),
                .5 * np.einsum('rw, ls, nkpqtu->nwlkpqrstu', *[np.eye(self.ncas_l2)]*2,
                               self.three_rdm_l2),
                - .5 * np.einsum('pw, nqlkrstu->nwlkpqrstu', np.eye(self.ncas_l2),
                                 self.four_rdm_l2),
                .5 * np.einsum('pw, lq, nkrstu->nwlkpqrstu', *[np.eye(self.ncas_l2)]*2,
                               self.three_rdm_l2),
                - .5 * np.einsum('lw, nkpqrstu->nwlkpqrstu', np.eye(self.ncas_l2),
                                 self.four_rdm_l2),
            ))

    def four_ct2_rdm(self, fragment):
        r"""
        .. math::
            \langle a_{m\sigma} E_{lk} E_{pq} E_{rs} E_{tu} a_{v\sigma}^\dagger \rangle
        """
        if fragment == 'l1':
            return sum((
                np.einsum('vm, lkpqrstu->vmlkpqrstu', np.eye(self.ncas_l1),
                          self.four_rdm_l1),
                - .5 * self.five_rdm_l1,
                np.einsum('uv, tmlkpqrs->vmlkpqrstu', np.eye(self.ncas_l1),
                          self.three_ct2_rdm('l1')),
                np.einsum('sv, rm, lkpqtu->vmlkpqrstu', *[np.eye(self.ncas_l1)]*2,
                          self.three_rdm_l1),
                - .5 * np.einsum('sv, rmlkpqtu->vmlkpqrstu', np.eye(self.ncas_l1),
                                 self.four_rdm_l1),
                np.einsum('sv, rq, pm, lktu->vmlkpqrstu', *[np.eye(self.ncas_l1)]*3,
                          self.two_rdm_l1_ord),
                - .5 * np.einsum('sv, rq, pmlktu->vmlkpqrstu', np.eye(self.ncas_l1),
                                 np.eye(self.ncas_l1), self.three_rdm_l1),
                np.einsum('sv, rk, lm, pqtu->vmlkpqrstu', *[np.eye(self.ncas_l1)]*3,
                          self.two_rdm_l1_ord),
                - .5 * np.einsum('sv, rk, lmpqtu->vmlkpqrstu', np.eye(self.ncas_l1),
                                 np.eye(self.ncas_l1), self.three_rdm_l1),
                np.einsum('sv, rq, pk, lm, tu->vmlkpqrstu', *[np.eye(self.ncas_l1)]*4,
                          self.one_rdm_l1),
                - .5 * np.einsum('sv, rq, pk, lmtu->vmlkpqrstu', *[np.eye(self.ncas_l1)]*3,
                                 self.two_rdm_l1_ord),
                np.einsum('qv, pm, lkrstu->vmlkpqrstu', *[np.eye(self.ncas_l1)]*2,
                          self.three_rdm_l1),
                - .5 * np.einsum('qv, pmlkrstu->vmlkpqrstu', np.eye(self.ncas_l1),
                                 self.four_rdm_l1),
                np.einsum('qv, pk, lm, rstu->vmlkpqrstu', *[np.eye(self.ncas_l1)]*3,
                          self.two_rdm_l1_ord),
                - .5 * np.einsum('qv, pk, lmrstu->vmlkpqrstu', *[np.eye(self.ncas_l1)]*2,
                                 self.three_rdm_l1),
                np.einsum('kv, lm, pqrstu->vmlkpqrstu', *[np.eye(self.ncas_l1)]*2,
                          self.three_rdm_l1),
                - .5 * np.einsum('kv, lmpqrstu->vmlkpqrstu', np.eye(self.ncas_l1),
                                 self.four_rdm_l1),
            ))
        elif fragment == 'l2':
            return sum((
                np.einsum('vm, lkpqrstu->vmlkpqrstu', np.eye(self.ncas_l2),
                          self.four_rdm_l2),
                - .5 * self.five_rdm_l2,
                np.einsum('uv, tmlkpqrs->vmlkpqrstu', np.eye(self.ncas_l2),
                          self.three_ct2_rdm('l2')),
                np.einsum('sv, rm, lkpqtu->vmlkpqrstu', *[np.eye(self.ncas_l2)]*2,
                          self.three_rdm_l2),
                - .5 * np.einsum('sv, rmlkpqtu->vmlkpqrstu', np.eye(self.ncas_l2),
                                 self.four_rdm_l2),
                np.einsum('sv, rq, pm, lktu->vmlkpqrstu', *[np.eye(self.ncas_l2)]*3,
                          self.two_rdm_l2_ord),
                - .5 * np.einsum('sv, rq, pmlktu->vmlkpqrstu', np.eye(self.ncas_l2),
                                 np.eye(self.ncas_l2), self.three_rdm_l2),
                np.einsum('sv, rk, lm, pqtu->vmlkpqrstu', *[np.eye(self.ncas_l2)]*3,
                          self.two_rdm_l2_ord),
                - .5 * np.einsum('sv, rk, lmpqtu->vmlkpqrstu', np.eye(self.ncas_l2),
                                 np.eye(self.ncas_l2), self.three_rdm_l2),
                np.einsum('sv, rq, pk, lm, tu->vmlkpqrstu', *[np.eye(self.ncas_l2)]*4,
                          self.one_rdm_l2),
                - .5 * np.einsum('sv, rq, pk, lmtu->vmlkpqrstu', *[np.eye(self.ncas_l2)]*3,
                                 self.two_rdm_l2_ord),
                np.einsum('qv, pm, lkrstu->vmlkpqrstu', *[np.eye(self.ncas_l2)]*2,
                          self.three_rdm_l2),
                - .5 * np.einsum('qv, pmlkrstu->vmlkpqrstu', np.eye(self.ncas_l2),
                                 self.four_rdm_l2),
                np.einsum('qv, pk, lm, rstu->vmlkpqrstu', *[np.eye(self.ncas_l2)]*3,
                          self.two_rdm_l2_ord),
                - .5 * np.einsum('qv, pk, lmrstu->vmlkpqrstu', *[np.eye(self.ncas_l2)]*2,
                                 self.three_rdm_l2),
                np.einsum('kv, lm, pqrstu->vmlkpqrstu', *[np.eye(self.ncas_l2)]*2,
                          self.three_rdm_l2),
                - .5 * np.einsum('kv, lmpqrstu->vmlkpqrstu', np.eye(self.ncas_l2),
                                 self.four_rdm_l2),
            ))

    def get_ct_frag_rdms_utf(self, ct):
        if ct == 'A':
            zero_ct_rdm_l1 = self.zero_ct2_rdm('l1')
            one_ct_rdm_l1 = self.one_ct2_rdm('l1')
            two_ct_rdm_l1 = self.two_ct2_rdm('l1')
            three_ct_rdm_l1 = self.three_ct2_rdm('l1')
            four_ct_rdm_l1 = self.four_ct2_rdm('l1')

            zero_ct_rdm_l2 = .5 * self.one_rdm_l2
            one_ct_rdm_l2 = self.one_ct1_rdm('l2')
            two_ct_rdm_l2 = self.two_ct1_rdm('l2')
            three_ct_rdm_l2 = self.three_ct1_rdm('l2')
            four_ct_rdm_l2 = self.four_ct1_rdm('l2')
        elif ct == 'B':
            zero_ct_rdm_l1 = .5 * self.one_rdm_l1
            one_ct_rdm_l1 = self.one_ct1_rdm('l1')
            two_ct_rdm_l1 = self.two_ct1_rdm('l1')
            three_ct_rdm_l1 = self.three_ct1_rdm('l1')
            four_ct_rdm_l1 = self.four_ct1_rdm('l1')

            zero_ct_rdm_l2 = self.zero_ct2_rdm('l2')
            one_ct_rdm_l2 = self.one_ct2_rdm('l2')
            two_ct_rdm_l2 = self.two_ct2_rdm('l2')
            three_ct_rdm_l2 = self.three_ct2_rdm('l2')
            four_ct_rdm_l2 = self.four_ct2_rdm('l2')

        return (zero_ct_rdm_l1, one_ct_rdm_l1, two_ct_rdm_l1, three_ct_rdm_l1, four_ct_rdm_l1,
                zero_ct_rdm_l2, one_ct_rdm_l2, two_ct_rdm_l2, three_ct_rdm_l2, four_ct_rdm_l2)

    def get_h0_pure_ct(self, fragment, ct, localex_l, localex_r):
        (zero_ct_rdm_l1, one_ct_rdm_l1, two_ct_rdm_l1, three_ct_rdm_l1, four_ct_rdm_l1,
         zero_ct_rdm_l2, one_ct_rdm_l2, two_ct_rdm_l2, three_ct_rdm_l2,
         four_ct_rdm_l2) = self.get_ct_frag_rdms_utf(ct)

        if fragment == 'l1':
            c1_pure = self.c1[np.ix_(*[self.l1_idx]*2)]
            c2_pure = self.c2[np.ix_(*[self.l1_idx]*4)]
            c1 = h_prime(c1_pure, c2_pure)

            if ct == 'A':
                if localex_l == 'A' and localex_r == 'A':
                    h1 = 2 * np.einsum('pq, vmlkpqtu, nw->klmntuvw', c1, three_ct_rdm_l1,
                                       zero_ct_rdm_l2)
                    h2 = np.einsum('pqrs, vmlkpqrstu, nw->klmntuvw',
                                   c2_pure, four_ct_rdm_l1, zero_ct_rdm_l2)
                elif localex_l == 'A' and localex_r == 'B':
                    h1 = 2 * np.einsum('pq, vmlkpq, nwtu->klmntuvw', c1, two_ct_rdm_l1,
                                       one_ct_rdm_l2)
                    h2 = np.einsum('pqrs, vmlkpqrs, nwtu->klmntuvw',
                                   c2_pure, three_ct_rdm_l1, one_ct_rdm_l2)
                elif localex_l == 'B' and localex_r == 'A':
                    h1 = 2 * np.einsum('pq, vmpqtu, nwlk->klmntuvw', c1, two_ct_rdm_l1,
                                       one_ct_rdm_l2)
                    h2 = np.einsum('pqrs, vmpqrstu, nwlk->klmntuvw',
                                   c2_pure, three_ct_rdm_l1, one_ct_rdm_l2)
                elif localex_l == 'B' and localex_r == 'B':
                    h1 = 2 * np.einsum('pq, vmpq, nwlktu->klmntuvw', c1, one_ct_rdm_l1,
                                       two_ct_rdm_l2)
                    h2 = np.einsum('pqrs, vmpqrs, nwlktu->klmntuvw',
                                   c2_pure, two_ct_rdm_l1, two_ct_rdm_l2)
            elif ct == 'B':
                if localex_l == 'A' and localex_r == 'A':
                    h1 = 2 * np.einsum('pq, nwlkpqtu, vm->klmntuvw', c1, three_ct_rdm_l1,
                                       zero_ct_rdm_l2)
                    h2 = np.einsum('pqrs, nwlkpqrstu, vm->klmntuvw',
                                   c2_pure, four_ct_rdm_l1, zero_ct_rdm_l2)
                elif localex_l == 'A' and localex_r == 'B':
                    h1 = 2 * np.einsum('pq, nwlkpq, vmtu->klmntuvw', c1, two_ct_rdm_l1,
                                       one_ct_rdm_l2)
                    h2 = np.einsum('pqrs, nwlkpqrs, vmtu->klmntuvw',
                                   c2_pure, three_ct_rdm_l1, one_ct_rdm_l2)
                elif localex_l == 'B' and localex_r == 'A':
                    h1 = 2 * np.einsum('pq, nwpqtu, vmlk->klmntuvw', c1, two_ct_rdm_l1,
                                       one_ct_rdm_l2)
                    h2 = np.einsum('pqrs, nwpqrstu, vmlk->klmntuvw',
                                   c2_pure, three_ct_rdm_l1, one_ct_rdm_l2)
                elif localex_l == 'B' and localex_r == 'B':
                    h1 = 2 * np.einsum('pq, nwpq, vmlktu->klmntuvw', c1, one_ct_rdm_l1,
                                       two_ct_rdm_l2)
                    h2 = np.einsum('pqrs, nwpqrs, vmlktu->klmntuvw',
                                   c2_pure, two_ct_rdm_l1, two_ct_rdm_l2)
        elif fragment == 'l2':
            c1_pure = self.c1[np.ix_(*[self.l2_idx]*2)]
            c2_pure = self.c2[np.ix_(*[self.l2_idx]*4)]
            c1 = h_prime(c1_pure, c2_pure)

            if ct == 'A':
                if localex_l == 'A' and localex_r == 'A':
                    h1 = 2 * np.einsum('pq, vmlktu, nwpq->klmntuvw', c1, two_ct_rdm_l1,
                                       one_ct_rdm_l2)
                    h2 = np.einsum('pqrs, vmlktu, nwpqrs->klmntuvw',
                                   c2_pure, two_ct_rdm_l1, two_ct_rdm_l2)
                elif localex_l == 'A' and localex_r == 'B':
                    h1 = 2 * np.einsum('pq, vmlk, nwpqtu->klmntuvw', c1, one_ct_rdm_l1,
                                       two_ct_rdm_l2)
                    h2 = np.einsum('pqrs, vmlk, nwpqrstu->klmntuvw',
                                   c2_pure, one_ct_rdm_l1, three_ct_rdm_l2)
                elif localex_l == 'B' and localex_r == 'A':
                    h1 = 2 * np.einsum('pq, vmtu, nwlkpq->klmntuvw', c1, one_ct_rdm_l1,
                                       two_ct_rdm_l2)
                    h2 = np.einsum('pqrs, vmtu, nwlkpqrs->klmntuvw',
                                   c2_pure, one_ct_rdm_l1, three_ct_rdm_l2)
                elif localex_l == 'B' and localex_r == 'B':
                    h1 = 2 * np.einsum('pq, vm, nwlkpqtu->klmntuvw', c1, zero_ct_rdm_l1,
                                       three_ct_rdm_l2)
                    h2 = np.einsum('pqrs, vm, nwlkpqrstu->klmntuvw',
                                   c2_pure, zero_ct_rdm_l1, four_ct_rdm_l2)
            elif ct == 'B':
                if localex_l == 'A' and localex_r == 'A':
                    h1 = 2 * np.einsum('pq, nwlktu, vmpq->klmntuvw', c1, two_ct_rdm_l1,
                                       one_ct_rdm_l2)
                    h2 = np.einsum('pqrs, nwlktu, vmpqrs->klmntuvw',
                                   c2_pure, two_ct_rdm_l1, two_ct_rdm_l2)
                elif localex_l == 'A' and localex_r == 'B':
                    h1 = 2 * np.einsum('pq, nwlk, vmpqtu->klmntuvw', c1, one_ct_rdm_l1,
                                       two_ct_rdm_l2)
                    h2 = np.einsum('pqrs, nwlk, vmpqrstu->klmntuvw',
                                   c2_pure, one_ct_rdm_l1, three_ct_rdm_l2)
                elif localex_l == 'B' and localex_r == 'A':
                    h1 = 2 * np.einsum('pq, nwtu, vmlkpq->klmntuvw', c1, one_ct_rdm_l1,
                                       two_ct_rdm_l2)
                    h2 = np.einsum('pqrs, nwtu, vmlkpqrs->klmntuvw',
                                   c2_pure, one_ct_rdm_l1, three_ct_rdm_l2)
                elif localex_l == 'B' and localex_r == 'B':
                    h1 = 2 * np.einsum('pq, nw, vmlkpqtu->klmntuvw', c1, zero_ct_rdm_l1,
                                       three_ct_rdm_l2)
                    h2 = np.einsum('pqrs, nw, vmlkpqrstu->klmntuvw',
                                   c2_pure, zero_ct_rdm_l1, four_ct_rdm_l2)
        return h1 + h2

    def get_h0_disp_ct(self, ct, localex_l, localex_r):
        (zero_ct_rdm_l1, one_ct_rdm_l1, two_ct_rdm_l1, three_ct_rdm_l1, four_ct_rdm_l1,
         zero_ct_rdm_l2, one_ct_rdm_l2, two_ct_rdm_l2, three_ct_rdm_l2,
         four_ct_rdm_l2) = self.get_ct_frag_rdms_utf(ct)

        gprime = self.exchange
        if ct == 'A':
            if localex_l == 'A' and localex_r == 'A':
                h1 = 2 * np.einsum('pqrs, rs, vmlkpqtu, nw->klmntuvw', gprime,
                                   self.one_rdm_l2, three_ct_rdm_l1, zero_ct_rdm_l2)
                h2 = 2 * np.einsum('pqrs, pq, vmlktu, nwrs->klmntuvw', gprime,
                                   self.one_rdm_l1, two_ct_rdm_l1, one_ct_rdm_l2)
            elif localex_l == 'A' and localex_r == 'B':
                h1 = 2 * np.einsum('pqrs, rs, vmlkpq, nwtu->klmntuvw', gprime,
                                   self.one_rdm_l2, two_ct_rdm_l1, one_ct_rdm_l2)
                h2 = 2 * np.einsum('pqrs, pq, vmlk, nwrstu->klmntuvw', gprime,
                                   self.one_rdm_l1, one_ct_rdm_l1, two_ct_rdm_l2)
            elif localex_l == 'B' and localex_r == 'A':
                h1 = 2 * np.einsum('pqrs, rs, vmpqtu, nwlk->klmntuvw', gprime,
                                   self.one_rdm_l2, two_ct_rdm_l1, one_ct_rdm_l2)
                h2 = 2 * np.einsum('pqrs, pq, vmtu, nwlkrs->klmntuvw', gprime,
                                   self.one_rdm_l1, one_ct_rdm_l1, two_ct_rdm_l2)
            elif localex_l == 'B' and localex_r == 'B':
                h1 = 2 * np.einsum('pqrs, rs, vmpq, nwlktu->klmntuvw', gprime,
                                   self.one_rdm_l2, one_ct_rdm_l1, two_ct_rdm_l2)
                h2 = 2 * np.einsum('pqrs, pq, vm, nwlkrstu->klmntuvw', gprime,
                                   self.one_rdm_l1, zero_ct_rdm_l1, three_ct_rdm_l2)
        if ct == 'B':
            if localex_l == 'A' and localex_r == 'A':
                h1 = 2 * np.einsum('pqrs, rs, nwlkpqtu, vm->klmntuvw', gprime,
                                   self.one_rdm_l2, three_ct_rdm_l1, zero_ct_rdm_l2)
                h2 = 2 * np.einsum('pqrs, pq, nwlktu, vmrs->klmntuvw', gprime,
                                   self.one_rdm_l1, two_ct_rdm_l1, one_ct_rdm_l2)
            elif localex_l == 'A' and localex_r == 'B':
                h1 = 2 * np.einsum('pqrs, rs, nwlkpq, vmtu->klmntuvw', gprime,
                                   self.one_rdm_l2, two_ct_rdm_l1, one_ct_rdm_l2)
                h2 = 2 * np.einsum('pqrs, pq, nwlk, vmrstu->klmntuvw', gprime,
                                   self.one_rdm_l1, one_ct_rdm_l1, two_ct_rdm_l2)
            elif localex_l == 'B' and localex_r == 'A':
                h1 = 2 * np.einsum('pqrs, rs, nwpqtu, vmlk->klmntuvw', gprime,
                                   self.one_rdm_l2, two_ct_rdm_l1, one_ct_rdm_l2)
                h2 = 2 * np.einsum('pqrs, pq, nwtu, vmlkrs->klmntuvw', gprime,
                                   self.one_rdm_l1, one_ct_rdm_l1, two_ct_rdm_l2)
            elif localex_l == 'B' and localex_r == 'B':
                h1 = 2 * np.einsum('pqrs, rs, nwpq, vmlktu->klmntuvw', gprime,
                                   self.one_rdm_l2, one_ct_rdm_l1, two_ct_rdm_l2)
                h2 = 2 * np.einsum('pqrs, pq, nw, vmlkrstu->klmntuvw', gprime,
                                   self.one_rdm_l1, zero_ct_rdm_l1, three_ct_rdm_l2)
        h3 = - np.einsum('pqrs, pq, rs', gprime,
                         self.one_rdm_l1, self.one_rdm_l2) * self.get_overlap_ct(
                             ct, localex_l, localex_r)
        return h1 + h2 + h3

    def get_overlap_ct(self, ct, localex_l, localex_r):
        (zero_ct_rdm_l1, one_ct_rdm_l1, two_ct_rdm_l1,
         zero_ct_rdm_l2, one_ct_rdm_l2, two_ct_rdm_l2) = self.get_ct_frag_rdms(ct)
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

    def get_ham_ct_col(self, ct, localex):
        (zero_ct_rdm_l1, one_ct_rdm_l1, two_ct_rdm_l1,
         zero_ct_rdm_l2, one_ct_rdm_l2, two_ct_rdm_l2) = self.get_ct_frag_rdms(ct)

        g_eff_A = self.c2[np.ix_(
            self.l1_idx, self.l1_idx, self.l1_idx, self.l2_idx)]
        g_eff_B = self.c2[np.ix_(
            self.l2_idx, self.l2_idx, self.l2_idx, self.l1_idx)]

        if ct == 'A':
            h_eff = self.c1[np.ix_(self.l1_idx, self.l2_idx)] - np.einsum(
                'prrq->pq', self.c2[np.ix_(self.l1_idx, self.l1_idx, self.l1_idx, self.l2_idx)])
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
            h_eff = self.c1[np.ix_(self.l2_idx, self.l1_idx)] - np.einsum(
                'prrq->pq', self.c2[np.ix_(self.l2_idx, self.l2_idx, self.l2_idx, self.l1_idx)])
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

    def compute_all_ct_matrices(self, ct):
        nex_AA = self.ncas_l1 * self.ncas_l1 * self.ncas_l1 * self.ncas_l2
        nex_BB = self.ncas_l2 * self.ncas_l2 * self.ncas_l1 * self.ncas_l2

        S_ovlp_AA_AA = self.get_overlap_ct(
            ct, 'A', 'A').reshape((nex_AA, nex_AA))
        S_ovlp_AA_BB = self.get_overlap_ct(
            ct, 'A', 'B').reshape((nex_AA, nex_BB))
        S_ovlp_BB_AA = self.get_overlap_ct(
            ct, 'B', 'A').reshape((nex_BB, nex_AA))
        S_ovlp_BB_BB = self.get_overlap_ct(
            ct, 'B', 'B').reshape((nex_BB, nex_BB))

        H_l1_AA_AA = self.get_h0_pure_ct(
            'l1', ct, 'A', 'A').reshape((nex_AA, nex_AA))
        H_l1_AA_BB = self.get_h0_pure_ct(
            'l1', ct, 'A', 'B').reshape((nex_AA, nex_BB))
        H_l1_BB_AA = self.get_h0_pure_ct(
            'l1', ct, 'B', 'A').reshape((nex_BB, nex_AA))
        H_l1_BB_BB = self.get_h0_pure_ct(
            'l1', ct, 'B', 'B').reshape((nex_BB, nex_BB))

        H_l2_AA_AA = self.get_h0_pure_ct(
            'l2', ct, 'A', 'A').reshape((nex_AA, nex_AA))
        H_l2_AA_BB = self.get_h0_pure_ct(
            'l2', ct, 'A', 'B').reshape((nex_AA, nex_BB))
        H_l2_BB_AA = self.get_h0_pure_ct(
            'l2', ct, 'B', 'A').reshape((nex_BB, nex_AA))
        H_l2_BB_BB = self.get_h0_pure_ct(
            'l2', ct, 'B', 'B').reshape((nex_BB, nex_BB))

        H_0_ct_AA_AA = self.get_h0_disp_ct(
            ct, 'A', 'A').reshape((nex_AA, nex_AA))
        H_0_ct_AA_BB = self.get_h0_disp_ct(
            ct, 'A', 'B').reshape((nex_AA, nex_BB))
        H_0_ct_BB_AA = self.get_h0_disp_ct(
            ct, 'B', 'A').reshape((nex_BB, nex_AA))
        H_0_ct_BB_BB = self.get_h0_disp_ct(
            ct, 'B', 'B').reshape((nex_BB, nex_BB))

        S_ovlp = np.block([[S_ovlp_AA_AA, S_ovlp_AA_BB],
                           [S_ovlp_BB_AA, S_ovlp_BB_BB]])
        H_l1 = np.block([[H_l1_AA_AA, H_l1_AA_BB],
                         [H_l1_BB_AA, H_l1_BB_BB]])
        H_l2 = np.block([[H_l2_AA_AA, H_l2_AA_BB],
                         [H_l2_BB_AA, H_l2_BB_BB]])
        H_0_ct = np.block([[H_0_ct_AA_AA, H_0_ct_AA_BB],
                           [H_0_ct_BB_AA, H_0_ct_BB_BB]])

        H_0 = H_l1 + H_l2 + H_0_ct

        # print(f'Hermitian check ct={ct}, should be true:')
        # print(np.allclose(S_ovlp, S_ovlp.T))
        # print(np.allclose(H_l1, H_l1.T))
        # print(np.allclose(H_l2, H_l2.T))
        # print(np.allclose(H_0_ct, H_0_ct.T))
        # print(np.allclose(H_0, H_0.T))

        return S_ovlp, H_0

    def run_fragpt2_ct(self):
        self.get_four_rdms()
        self.get_five_rdms()

        E_0 = self.h0_expval(w_core=False)

        # -------- AB ----------
        S_ovlp_AB, H_0_AB = self.compute_all_ct_matrices('A')

        nex_AAAB = self.ncas_l1 * self.ncas_l1 * self.ncas_l1 * self.ncas_l2
        nex_BBAB = self.ncas_l2 * self.ncas_l2 * self.ncas_l1 * self.ncas_l2

        H_prime_col_AAAB = self.get_ham_ct_col('A', 'A')
        H_prime_col_AAAB = H_prime_col_AAAB.reshape((nex_AAAB))
        H_prime_col_BBAB = self.get_ham_ct_col('A', 'B')
        H_prime_col_BBAB = H_prime_col_BBAB.reshape((nex_BBAB))
        H_prime_col_AB = np.concatenate((H_prime_col_AAAB, H_prime_col_BBAB))

        self.psi1_c_ct_AB = np.linalg.solve((H_0_AB - E_0 * S_ovlp_AB),
                                            - H_prime_col_AB)
        e_pt2_AB = np.einsum('n, n', H_prime_col_AB,
                             self.psi1_c_ct_AB)

        # -------- BA ----------
        S_ovlp_BA, H_0_BA = self.compute_all_ct_matrices('B')

        nex_AABA = self.ncas_l1 * self.ncas_l1 * self.ncas_l2 * self.ncas_l1
        nex_BBBA = self.ncas_l2 * self.ncas_l2 * self.ncas_l2 * self.ncas_l1

        H_prime_col_AABA = self.get_ham_ct_col('B', 'A')
        H_prime_col_AABA = H_prime_col_AABA.reshape((nex_AABA))
        H_prime_col_BBBA = self.get_ham_ct_col('B', 'B')
        H_prime_col_BBBA = H_prime_col_BBBA.reshape((nex_BBBA))
        H_prime_col_BA = np.concatenate((H_prime_col_AABA, H_prime_col_BBBA))

        self.psi1_c_ct_BA = np.linalg.solve((H_0_BA - E_0 * S_ovlp_BA),
                                            - H_prime_col_BA)

        e_pt2_BA = np.einsum('n, n', H_prime_col_BA,
                             self.psi1_c_ct_BA)

        return e_pt2_AB, e_pt2_BA

    def two_tt_rdm(self, fragment):
        r"""
        For fragment A:

        .. math::
            \langle \Psi_A | T^{(1,0)}_{l k} E_{pq} E_{rs} T^{(1,0)}_{tu} | \Psi_A \rangle
            \langle \Psi_B | T^{(1,0)}_{n m}  T^{(1,0)}_{vw} | \Psi_B \rangle  \\
            + \langle \Psi_A | {T^{(1,1)}_{l k} E_{pq} E_{rs} T^{(1,-1)}_{tu} | \Psi_A \rangle
            \langle \Psi_B | T^{(1,-1)}_{n m} T^{(1,1)}_{vw} | \Psi_B} \rangle \\
            + \langle \Psi_A | T^{(1,-1)}_{l k} E_{pq} E_{rs} T^{(1,1)}_{tu} | \Psi_A \rangle
            \langle \Psi_B | T^{(1,1)}_{n m} T^{(1,-1)}_{vw} | \Psi_B \rangle \bigg)

        for fragment B:

        .. math::
            \langle \Psi_A | T^{(1,0)}_{l k}  T^{(1,0)}_{tu} | \Psi_A \rangle
            \langle \Psi_B | T^{(1,0)}_{n m} E_{pq} E_{rs} T^{(1,0)}_{vw} | \Psi_B \rangle\\
            + \langle \Psi_A | {T^{(1,1)}_{l k}  T^{(1,-1)}_{tu} | \Psi_A \rangle
            \langle \Psi_B | T^{(1,-1)}_{n m} E_{pq} E_{rs} T^{(1,1)}_{vw} | \Psi_B} \rangle\\
            + \langle \Psi_A | T^{(1,-1)}_{l k}  T^{(1,1)}_{tu} | \Psi_A \rangle
            \langle \Psi_B | T^{(1,1)}_{n m} E_{pq} E_{rs} T^{(1,-1)}_{vw} | \Psi_B \rangle \bigg)

        """
        if fragment == 'l1':
            return sum((
                np.einsum('lkpqrstu, nmvw -> klmnpqrstuvw',
                          self.four_rdm_aabb_l1, self.two_rdm_aabb_l2),
                np.einsum('lkpqrstu, nmvw -> klmnpqrstuvw',
                          self.four_rdm_abba_l1, self.two_rdm_baab_l2),
                np.einsum('lkpqrstu, nmvw -> klmnpqrstuvw',
                          self.four_rdm_baab_l1, self.two_rdm_abba_l2)
            ))
        elif fragment == 'l2':
            return sum((
                np.einsum('lktu, nmpqrsvw -> klmnpqrstuvw',
                          self.two_rdm_aabb_l1, self.four_rdm_aabb_l2),
                np.einsum('lktu, nmpqrsvw -> klmnpqrstuvw',
                          self.two_rdm_abba_l1, self.four_rdm_baab_l2),
                np.einsum('lktu, nmpqrsvw -> klmnpqrstuvw',
                          self.two_rdm_baab_l1, self.four_rdm_abba_l2)
            ))

    def one_tt_rdm(self, fragment):
        r"""
        For fragment A:

        .. math::
            \langle \Psi_A | T^{(1,0)}_{l k} E_{pq} T^{(1,0)}_{tu} | \Psi_A \rangle
            \langle \Psi_B | T^{(1,0)}_{n m}  T^{(1,0)}_{vw} | \Psi_B \rangle\\
            + \langle \Psi_A | {T^{(1,1)}_{l k} E_{pq} T^{(1,-1)}_{tu} | \Psi_A \rangle
            \langle \Psi_B | T^{(1,-1)}_{n m} T^{(1,1)}_{vw} | \Psi_B} \rangle\\
            + \langle \Psi_A | T^{(1,-1)}_{l k} E_{pq} T^{(1,1)}_{tu} | \Psi_A \rangle
            \langle \Psi_B | T^{(1,1)}_{n m} T^{(1,-1)}_{vw} | \Psi_B \rangle \bigg)

        for fragment B:

        .. math::
            \langle \Psi_A | T^{(1,0)}_{l k}  T^{(1,0)}_{tu} | \Psi_A \rangle
            \langle \Psi_B | T^{(1,0)}_{n m} E_{pq} T^{(1,0)}_{vw} | \Psi_B \rangle\\
            + \langle \Psi_A | {T^{(1,1)}_{l k}  T^{(1,-1)}_{tu} | \Psi_A \rangle
            \langle \Psi_B | T^{(1,-1)}_{n m} E_{pq} T^{(1,1)}_{vw} | \Psi_B} \rangle\\
            + \langle \Psi_A | T^{(1,-1)}_{l k}  T^{(1,1)}_{tu} | \Psi_A \rangle
            \langle \Psi_B | T^{(1,1)}_{n m} E_{pq} T^{(1,-1)}_{vw} | \Psi_B \rangle \bigg)

        """
        if fragment == 'l1':
            return sum((
                np.einsum('lkpqtu, nmvw -> klmnpqtuvw',
                          self.three_rdm_aabb_l1, self.two_rdm_aabb_l2),
                np.einsum('lkpqtu, nmvw -> klmnpqtuvw',
                          self.three_rdm_abba_l1, self.two_rdm_baab_l2),
                np.einsum('lkpqtu, nmvw -> klmnpqtuvw',
                          self.three_rdm_baab_l1, self.two_rdm_abba_l2)
            ))
        elif fragment == 'l2':
            return sum((
                np.einsum('lktu, nmpqvw -> klmnpqtuvw',
                          self.two_rdm_aabb_l1, self.three_rdm_aabb_l2),
                np.einsum('lktu, nmpqvw -> klmnpqtuvw',
                          self.two_rdm_abba_l1, self.three_rdm_baab_l2),
                np.einsum('lktu, nmpqvw -> klmnpqtuvw',
                          self.two_rdm_baab_l1, self.three_rdm_abba_l2)
            ))

    def zero_tt_rdm(self):
        r"""

        .. math::
            \langle \Psi_A | T^{(1,0)}_{l k} T^{(1,0)}_{tu} | \Psi_A \rangle
            \langle \Psi_B | T^{(1,0)}_{n m}  T^{(1,0)}_{vw} | \Psi_B \rangle\\
            + \langle \Psi_A | {T^{(1,1)}_{l k} T^{(1,-1)}_{tu} | \Psi_A \rangle
            \langle \Psi_B | T^{(1,-1)}_{n m} T^{(1,1)}_{vw} | \Psi_B} \rangle\\
            + \langle \Psi_A | T^{(1,-1)}_{l k} T^{(1,1)}_{tu} | \Psi_A \rangle
            \langle \Psi_B | T^{(1,1)}_{n m} T^{(1,-1)}_{vw} | \Psi_B \rangle \bigg)
        """
        return sum((
            np.einsum('lktu, nmvw -> klmntuvw',
                      self.two_rdm_aabb_l1, self.two_rdm_aabb_l2),
            np.einsum('lktu, nmvw -> klmntuvw',
                      self.two_rdm_abba_l1, self.two_rdm_baab_l2),
            np.einsum('lktu, nmvw -> klmntuvw',
                      self.two_rdm_baab_l1, self.two_rdm_abba_l2)
        ))

    def get_tt_rdms(self):
        if not self.got_tt_rdms:
            self.four_rdm_abba_l1 = make_dm4sabba(
                self.civec_l1, self.ncas_l1, self.nelec_l1)
            self.four_rdm_baab_l1 = make_dm4sbaab(
                self.civec_l1, self.ncas_l1, self.nelec_l1)
            self.four_rdm_aabb_l1 = make_dm4saabb(
                self.civec_l1, self.ncas_l1, self.nelec_l1)

            self.three_rdm_abba_l1 = make_dm3sabba(
                self.civec_l1, self.ncas_l1, self.nelec_l1)
            self.three_rdm_baab_l1 = make_dm3sbaab(
                self.civec_l1, self.ncas_l1, self.nelec_l1)
            self.three_rdm_aabb_l1 = make_dm3saabb(
                self.civec_l1, self.ncas_l1, self.nelec_l1)

            self.two_rdm_abba_l1, self.two_rdm_baab_l1, self.two_rdm_aabb_l1 = make_dm2s(
                self.civec_l1, self.ncas_l1, self.nelec_l1)

            self.four_rdm_abba_l2 = make_dm4sabba(
                self.civec_l2, self.ncas_l2, self.nelec_l2)
            self.four_rdm_baab_l2 = make_dm4sbaab(
                self.civec_l2, self.ncas_l2, self.nelec_l2)
            self.four_rdm_aabb_l2 = make_dm4saabb(
                self.civec_l2, self.ncas_l2, self.nelec_l2)

            self.three_rdm_abba_l2 = make_dm3sabba(
                self.civec_l2, self.ncas_l2, self.nelec_l2)
            self.three_rdm_baab_l2 = make_dm3sbaab(
                self.civec_l2, self.ncas_l2, self.nelec_l2)
            self.three_rdm_aabb_l2 = make_dm3saabb(
                self.civec_l2, self.ncas_l2, self.nelec_l2)

            self.two_rdm_abba_l2, self.two_rdm_baab_l2, self.two_rdm_aabb_l2 = make_dm2s(
                self.civec_l2, self.ncas_l2, self.nelec_l2)

            self.got_tt_rdms = True

    def get_h0_tt(self):

        c1_eff_l1 = self.c1[np.ix_(*[self.l1_idx]*2)]
        c2_eff_l1 = self.c2[np.ix_(*[self.l1_idx]*4)]

        c1_eff_l2 = self.c1[np.ix_(*[self.l2_idx]*2)]
        c2_eff_l2 = self.c2[np.ix_(*[self.l2_idx]*4)]

        h0_pure_one_l1 = np.einsum(
            'pq, klmnpqtuvw->klmntuvw', c1_eff_l1, self.one_tt_rdm('l1'))
        h0_pure_two_l1 = .5 * np.einsum('pqrs, klmnpqrstuvw->klmntuvw',
                                        c2_eff_l1, self.two_tt_rdm('l1'))

        h0_pure_one_l2 = np.einsum(
            'pq, klmnpqtuvw->klmntuvw', c1_eff_l2, self.one_tt_rdm('l2'))
        h0_pure_two_l2 = .5 * np.einsum('pqrs, klmnpqrstuvw->klmntuvw',
                                        c2_eff_l2, self.two_tt_rdm('l2'))

        h0_disp_0 = np.einsum('pqrs, rs, klmnpqtuvw->klmntuvw', self.exchange,
                              self.one_rdm_l2, self.one_tt_rdm('l1'))
        h0_disp_1 = np.einsum('pqrs, pq, klmnrstuvw->klmntuvw', self.exchange,
                              self.one_rdm_l1, self.one_tt_rdm('l2'))

        c_eff = np.einsum('pqrs,pqrs', self.exchange,
                          np.einsum('pq,rs->pqrs', self.one_rdm_l1, self.one_rdm_l2))
        h0_disp_2 = - c_eff * self.zero_tt_rdm()

        return sum((
            h0_pure_one_l1, h0_pure_two_l1, h0_pure_one_l2, h0_pure_two_l2,
            h0_disp_0, h0_disp_1, h0_disp_2))

    def get_h_tt_col(self):
        g_eff = self.c2[np.ix_(self.l1_idx, self.l2_idx,
                               self.l2_idx, self.l1_idx)]
        return - np.einsum('psrq, pqrstuvw->tuvw', g_eff, self.zero_tt_rdm())

    def run_fragpt2_tt(self):
        t1 = time.time()
        self.get_tt_rdms()
        print("computing TT rdms took:", time.time()-t1)
        E_0 = self.h0_expval(w_core=False)
        S_ovlp = self.zero_tt_rdm().reshape((self.nex, self.nex))
        H_0 = self.get_h0_tt()
        H_0 = H_0.reshape((self.nex, self.nex))

        H_prime_col = self.get_h_tt_col()

        H_prime_col = H_prime_col.reshape((self.nex))

        self.psi1_c_tt = np.linalg.solve((H_0 - E_0 * S_ovlp),
                                         -H_prime_col)

        e_pt2 = np.einsum('n, n', H_prime_col,
                          self.psi1_c_tt)
        return e_pt2


if __name__ == '__main__':
    from pyscf import gto, scf
    from fragpt2 import unpack_pyscf
    import matplotlib.pyplot as plt

    basis = 'unc-cc-pvdz'
    symmetry = 0
    cart = True

    mol = gto.Mole(atom='../../examples/water_ammonia/water-ammonia.xyz',
                   basis=basis, symmetry=symmetry, cart=cart)
    # mol = gto.Mole(atom='../../examples/biphenyl/dihedral_15.xyz',
    #                basis=basis, symmetry=symmetry, cart=cart)
    mol.max_memory = 8000

    mol.build()
    mf = scf.RHF(mol)
    mf.run()

    nao, mo_energies, mo_coeff = unpack_pyscf(
        "../../examples/water_ammonia/ribo.pyscf")
    # nao, mo_energies, mo_coeff = unpack_pyscf("../../examples/biphenyl/ribo_15.pyscf")

    # Water ammonia:
    # # for CAS(2,2)(2,2):
    # act_idx_l1_base1 = [4, 11]
    # act_idx_l2_base1 = [8, 27]

    # # for CAS(4,4)(4,4):
    act_idx_l1_base1 = [4, 5, 11, 12]
    act_idx_l2_base1 = [8, 9, 27, 28]

    # # # for CAS(2,2)(4,4):
    # act_idx_l1_base1 = [5, 11]
    # act_idx_l2_base1 = [8, 9, 27, 28]

    # # # for CAS(4,3)(4,4):
    # act_idx_l1_base1 = [4, 5, 11]
    # act_idx_l2_base1 = [8, 9, 27, 28]

    # # Biphenyl 15 deg:
    # # for CAS(8,8):
    # act_idx_l1_base1 = [26, 27, 42, 43]
    # act_idx_l2_base1 = [40, 41, 56, 57]

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

    # ncas = len(act_idx)
    # # nelecas = 6
    # nelec_l1 = 4
    # nelec_l2 = 4
    # ncas_l1 = 4
    # ncas_l2 = 4

    print(f'l1: CAS({nelec_l1},{ncas_l1})')
    print(f'l2: CAS({nelec_l2},{ncas_l2})')

    occ_idx = [x for x in range(mol.nelectron//2) if x not in act_idx]

    nroots = 1

    casci = FragPT2(mf, occ_idx, act_idx,
                    ((ncas_l1, nelec_l1), (ncas_l2, nelec_l2)),
                    mo_energies=mo_energies, mo_coeff=mo_coeff)

    casci.run_full_casci(nroots)
    casci.run_l1_casci(nroots)
    casci.run_l2_casci(nroots)
    casci.run_ct1_casci(nroots)
    casci.run_ct2_casci(nroots)
    e_tot, e_l1, e_l2 = casci.run_self_consistent_casci()

    e_corr_naive = casci.e_l1 + casci.e_l2 - 2 * mf.e_tot
    e_naive = casci.e_l1 + casci.e_l2 - mf.e_tot

    print(f'Naive energy:            {e_naive:.6f}')
    print(f'scf energy:              {e_tot:.6f}')
    print(f'exact energy:            {casci.e_full:.6f}')

    e_pt2 = casci.run_fragpt2()
    # e_pt3, e_pt2 = casci.run_fragpt3()

    # print(f'embedding energy w pt3:  {e_tot + e_pt2 + e_pt3:.6f}')
    print(f'embedding energy wo pt3: {e_tot + e_pt2:.6f}')
    print(f'Pt2 correction:          {e_pt2:.8f}')
    # print(f'Pt3 correction:          {e_pt3:.8f}')

    # e_gev = casci.solve_gev()

    # print(f'GEV problem energy: {e_gev:.6f}')

    # psi_c = casci.psi1_c[1:].reshape((casci.ncas_l1**2, casci.ncas_l2**2))

    # plt.imshow(psi_c)
    # plt.colorbar()
    # plt.show()

    # plt.imshow(psi_c_norm)
    # plt.colorbar()
    # plt.show()

    casci.run_ct1_casci(nroots)
    casci.run_ct2_casci(nroots)

    e_ct1_ci = casci.e_ct1 - mf.e_tot
    e_ct2_ci = casci.e_ct2 - mf.e_tot

    print(f'CT1 CI energy:      {e_ct1_ci:.6f}')
    print(f'CT2 CI energy:      {e_ct2_ci:.6f}')

    e_pt2_ct1, e_pt2_ct2 = casci.run_fragpt2_ct()

    print(f'CT1 PT2 energy:     {e_pt2_ct1:.6f}')
    print(f'CT2 PT2 energy:     {e_pt2_ct2:.6f}')

    print(
        f'Full embedding E:   {e_tot + e_pt2 + 2*(e_pt2_ct1 + e_pt2_ct2):.6f}')

    casci_full = Exact_1ct(mf, occ_idx, act_idx,
                           ((ncas_l1, nelec_l1), (ncas_l2, nelec_l2)),
                           mo_energies=mo_energies, mo_coeff=mo_coeff)

    casci_full.run_full_casci(nroots)
    casci_full.run_l1_casci(nroots)
    casci_full.run_l2_casci(nroots)
    casci_full.run_ct1_casci(nroots)
    casci_full.run_ct2_casci(nroots)
    e_tot, e_l1, e_l2 = casci_full.run_self_consistent_casci()

    # e_corr_naive = casci_full.e_l1 + casci_full.e_l2 - 2 * mf.e_tot
    # e_naive = casci_full.e_l1 + casci_full.e_l2 - mf.e_tot

    # print(f'Naive energy:            {e_naive:.6f}')
    # print(f'scf energy:              {e_tot:.6f}')
    # print(f'exact energy:            {casci_full.e_full:.6f}')

    e_pt2 = casci_full.run_fragpt2()
    # e_pt3, e_pt2 = casci.run_fragpt3()

    # print(f'embedding energy w pt3:  {e_tot + e_pt2 + e_pt3:.6f}')
    # print(f'embedding energy wo pt3: {e_tot + e_pt2:.6f}')
    # print(f'Pt2 correction:          {e_pt2:.8f}')
    # print(f'Pt3 correction:          {e_pt3:.8f}')

    # e_gev = casci_full.solve_gev()

    # print(f'GEV problem energy: {e_gev:.6f}')

    # psi_c = casci_full.psi1_c[1:].reshape((casci_full.ncas_l1**2, casci_full.ncas_l2**2))

    # plt.imshow(psi_c)
    # plt.colorbar()
    # plt.show()

    # plt.imshow(psi_c_norm)
    # plt.colorbar()
    # plt.show()

    # casci_full.run_ct1_casci(nroots)
    # casci_full.run_ct2_casci(nroots)

    # e_ct1_ci = casci_full.e_ct1 - mf.e_tot
    # e_ct2_ci = casci_full.e_ct2 - mf.e_tot

    # print(f'CT1 CI energy:      {e_ct1_ci:.6f}')
    # print(f'CT2 CI energy:      {e_ct2_ci:.6f}')

    e_pt2_ct1_full, e_pt2_ct2_full = casci_full.run_fragpt2_ct()

    print(f'CT1 PT2 WITH 5-RDM: {e_pt2_ct1_full:.6f}')
    print(f'CT2 PT2 WITH 5-RDM: {e_pt2_ct2_full:.6f}')

    print(
        f'embed E WITH 5-RDM: {e_tot + e_pt2 + (e_pt2_ct1_full + e_pt2_ct2_full):.6f}')

    e_pt2_tt = casci_full.run_fragpt2_tt()

    print(f'TT PT2 WITH 4-RDM: {e_pt2_tt:.6f}')

    print(f'embed E WITH 5-RDM and TT:',
          f' {e_tot + e_pt2 + (e_pt2_ct1_full + e_pt2_ct2_full) + e_pt2_tt:.6f}')
