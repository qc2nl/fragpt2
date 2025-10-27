#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:22:36 2023

@author: emielkoridon
"""


import numpy as np
import scipy
import time
from pyscf import scf, mcscf, ao2mo, fci
from pyscf.fci.direct_spin0 import trans_rdm12, trans_rdm1s, FCI, make_rdm1, make_rdm12
from pyscf.fci.rdm import make_dm1234
from pyscf.fci.addons import overlap
from fragpt2.fragci import FragCI


def h_prime(one_body_integrals, two_body_integrals):
    return one_body_integrals - .5 * np.einsum('prrq->pq', two_body_integrals)


class FragPT2(FragCI):
    """Child of FragCI class. Supports perturbation calculations of
    dispersion, charge-transfer and TT terms."""

    def __init__(self, mf: scf.hf.RHF,
                 occ_idx, act_idx,
                 fragments,
                 mo_energies=None, mo_coeff=None, verbose=0):
        """
        Initialize Perturbation class.

        Args:
            mf (scf.hf.RHF): Full RHF object from pyscf
            occ_idx (list): Occupied indices that are always kept doubly occupied
            act_idx (list): Active indices ordered by fragment, then occupied - virtual
            fragments (tuple): Tuple (name,ncas,nelecas) for each fragment
            mo_energies (np.ndarray, optional): MO energies
            mo_coeff (np.ndarray, optional): MO coefficients

        """
        super().__init__(mf, occ_idx, act_idx, fragments,
                         mo_energies=mo_energies, mo_coeff=mo_coeff, verbose=verbose)

        self.nex = self.ncas_l1 * self.ncas_l1 * self.ncas_l2 * self.ncas_l2

        self.three_rdm_l1 = None
        self.three_rdm_l2 = None
        self.four_rdm_l1 = None
        self.four_rdm_l2 = None

    def get_h0_disp_col(self):
        h1 = np.einsum('rs, pqtu, vw->pqrstuvw',
                       self.one_rdm_l2, self.two_rdm_l1_ord, self.one_rdm_l2)
        h2 = np.einsum('pq, tu, rsvw->pqrstuvw',
                       self.one_rdm_l1, self.one_rdm_l1, self.two_rdm_l2_ord)
        h3 = - np.einsum('pq, rs, tu, vw->pqrstuvw',
                         self.one_rdm_l1, self.one_rdm_l2,
                         self.one_rdm_l1, self.one_rdm_l2)
        return np.einsum('pqrs, pqrstuvw->tuvw', self.exchange, h1 + h2 + h3)

    def get_ham_disp_col(self):
        return np.einsum('pqrs, pqtu, rsvw->tuvw',
                         self.exchange, self.two_rdm_l1_ord, self.two_rdm_l2_ord)

    def get_wfn_overlap(self, w_zero=True):
        S_ovlp = np.einsum('lktu,nmvw->klmntuvw',
                           self.two_rdm_l1_ord, self.two_rdm_l2_ord).reshape((self.nex, self.nex))
        if w_zero:
            vec_to_add = np.einsum('tu, vw->tuvw',
                                   self.one_rdm_l1, self.one_rdm_l2).reshape((self.nex))
            S_ovlp = np.vstack((vec_to_add, S_ovlp))
            S_ovlp = np.hstack(
                (np.append(np.array([1.]), vec_to_add).reshape(self.nex+1, 1), S_ovlp))
        return S_ovlp

    def get_pure_ham_col(self, fragment):
        if fragment == 'l1':
            c1_pure = self.c1[np.ix_(*[self.l1_idx]*2)]
            c2_pure = self.c2[np.ix_(*[self.l1_idx]*4)]

            h1 = np.einsum('pq, pqtu->tu', c1_pure, self.two_rdm_l1_ord)
            h2_c = .5 * np.einsum('pqrs, pqrstu->tu',
                                  c2_pure, self.three_rdm_l1)
            h2_ex = -.5 * np.einsum('prrq, pqtu->tu',
                                    c2_pure, self.two_rdm_l1_ord)

            return np.einsum('tu, vw->tuvw', h1 + h2_c + h2_ex, self.one_rdm_l2)

        elif fragment == 'l2':
            c1_pure = self.c1[np.ix_(*[self.l2_idx]*2)]
            c2_pure = self.c2[np.ix_(*[self.l2_idx]*4)]

            h1 = np.einsum('pq, nmpq->mn', c1_pure, self.two_rdm_l2_ord)
            h2_c = .5 * np.einsum('pqrs, nmpqrs->mn',
                                  c2_pure, self.three_rdm_l2)
            h2_ex = -.5 * np.einsum('prrq, nmpq->mn',
                                    c2_pure, self.two_rdm_l2_ord)

            return np.einsum('lk, mn->klmn', self.one_rdm_l1, h1 + h2_c + h2_ex)
        else:
            raise ValueError(f'fragment {fragment} doesnt exist')

    def get_pure_ham(self, fragment):
        if fragment == 'l1':
            c1_pure = self.c1[np.ix_(*[self.l1_idx]*2)]
            c2_pure = self.c2[np.ix_(*[self.l1_idx]*4)]

            h1 = np.einsum('pq, lkpqtu->kltu', c1_pure, self.three_rdm_l1)
            h2_c = .5 * np.einsum('pqrs, lkpqrstu->kltu',
                                  c2_pure, self.four_rdm_l1)
            h2_ex = -.5 * np.einsum('prrq, lkpqtu->kltu',
                                    c2_pure, self.three_rdm_l1)

            return np.einsum('kltu, nmvw->klmntuvw', h1 + h2_c + h2_ex, self.two_rdm_l2_ord)

        elif fragment == 'l2':
            c1_pure = self.c1[np.ix_(*[self.l2_idx]*2)]
            c2_pure = self.c2[np.ix_(*[self.l2_idx]*4)]

            h1 = np.einsum('pq, nmpqvw->mnvw', c1_pure, self.three_rdm_l2)
            h2_c = .5 * np.einsum('pqrs, nmpqrsvw->mnvw',
                                  c2_pure, self.four_rdm_l2)
            h2_ex = -.5 * np.einsum('prrq, nmpqvw->mnvw',
                                    c2_pure, self.three_rdm_l2)

            return np.einsum('lktu, mnvw->klmntuvw', self.two_rdm_l1_ord, h1 + h2_c + h2_ex)
        else:
            raise ValueError(f'fragment {fragment} doesnt exist')

    def get_h0_disp(self):

        h1 = np.einsum('pqrs, rs, lkpqtu, nmvw->klmntuvw', self.exchange,
                       self.one_rdm_l2, self.three_rdm_l1, self.two_rdm_l2_ord)
        h2 = np.einsum('pqrs, pq, lktu, nmrsvw->klmntuvw', self.exchange,
                       self.one_rdm_l1, self.two_rdm_l1_ord, self.three_rdm_l2)
        h3 = - np.einsum('pqrs, pq, rs, lktu, nmvw->klmntuvw', self.exchange,
                         self.one_rdm_l1, self.one_rdm_l2, self.two_rdm_l1_ord, self.two_rdm_l2_ord)
        return h1 + h2 + h3

    def get_ham_disp(self):

        return np.einsum('pqrs, lkpqtu, nmrsvw->klmntuvw',
                         self.exchange, self.three_rdm_l1, self.three_rdm_l2)

    def get_four_rdms(self, again=False):
        if self.four_rdm_l1 is None or self.four_rdm_l2 is None or again:
            t1 = time.time()
            _, self.two_rdm_l1_ord, self.three_rdm_l1, self.four_rdm_l1 = make_dm1234(
                'FCI4pdm_kern_sf', self.civec_l1, self.civec_l1, self.ncas_l1, self.nelec_l1)
            _, self.two_rdm_l2_ord, self.three_rdm_l2, self.four_rdm_l2 = make_dm1234(
                'FCI4pdm_kern_sf', self.civec_l2, self.civec_l2, self.ncas_l2, self.nelec_l2)
            print('computing 4-RDMs took:', time.time()-t1)

    def run_fragpt2(self):
        self.get_four_rdms()
        E_0 = self.h0_expval(w_core=False)
        S_ovlp = self.get_wfn_overlap(w_zero=False)
        H_l1 = self.get_pure_ham('l1')  # .reshape((self.nex, self.nex))
        H_l2 = self.get_pure_ham('l2')  # .reshape((self.nex, self.nex))
        H_0_disp = self.get_h0_disp()  # .reshape((self.nex, self.nex))
        # H_disp = self.get_disp_ham().reshape((self.nex, self.nex))
        H_0 = H_l1 + H_l2 + H_0_disp
        H_0 = H_0.reshape((self.nex, self.nex))

        H_0_disp_col = self.get_h0_disp_col()

        # H_l1_col = self.get_pure_ham_col('l1')
        # H_l2_col = self.get_pure_ham_col('l2')

        # eigval, eigvec = scipy.linalg.eigh(H_0, b=S_ovlp)

        H_prime_col = (self.get_ham_disp_col() - H_0_disp_col)  # .reshape(
        # (self.nex))

        H_prime_col = H_prime_col.reshape((self.nex))
        # H_prime_col = np.append(np.array([0.]), H_prime_col)

        # H_prime_eig = eigvec.T @ H_prime_col

        self.psi1_c = np.linalg.solve((H_0 - E_0 * S_ovlp),
                                      -H_prime_col)

        e_pt2 = np.einsum('n, n', H_prime_col,
                          self.psi1_c)
        return e_pt2

    def run_fragpt3(self):
        e_pt2 = self.run_fragpt2()
        E_0 = self.h0_expval(w_core=False)
        S_ovlp = self.get_wfn_overlap(w_zero=False)
        S_ovlp_col = np.einsum('tu, vw->tuvw',
                               self.one_rdm_l1, self.one_rdm_l2).reshape((self.nex))

        H_l1 = self.get_pure_ham('l1')  # .reshape((self.nex, self.nex))
        H_l2 = self.get_pure_ham('l2')  # .reshape((self.nex, self.nex))
        H_0_disp = self.get_h0_disp()  # .reshape((self.nex, self.nex))
        # H_disp = self.get_disp_ham().reshape((self.nex, self.nex))
        H_0 = H_l1 + H_l2 + H_0_disp
        H_0 = H_0.reshape((self.nex, self.nex))

        H_0_disp_col = self.get_h0_disp_col()

        eigval, eigvec = scipy.linalg.eigh(H_0, b=S_ovlp)

        H_prime = self.get_ham_disp() - H_0_disp
        H_prime = H_prime.reshape((self.nex, self.nex))

        H_prime_col = (self.get_ham_disp_col() - H_0_disp_col)
        H_prime_col = H_prime_col.reshape((self.nex))

        H_prime_eig = eigvec.T @ H_prime @ eigvec
        H_prime_col_eig = eigvec.T @ H_prime_col

        S_ovlp_col_eig = eigvec.T @ S_ovlp_col

        assert np.allclose(E_0, eigval[0], atol=1e-6)

        norm = E_0 - eigval
        norm[0] = 1e99

        H_prime_col_norm = np.divide(H_prime_col_eig, norm)

        e_pt3 = sum((
            np.einsum('n, nm, m', H_prime_col_norm,
                      H_prime_eig, H_prime_col_norm),
            - e_pt2 * (H_prime_col_norm @ S_ovlp_col_eig)))
        return e_pt3, e_pt2

    def solve_gev(self):
        self.get_four_rdms()
        S_ovlp = self.get_wfn_overlap(w_zero=False)
        H_l1 = self.get_pure_ham('l1')
        H_l2 = self.get_pure_ham('l2')
        H_disp = self.get_ham_disp()
        H_full = H_l1 + H_l2 + H_disp
        H_full = H_full.reshape((self.nex, self.nex))

        v, w = scipy.linalg.eigh(H_full, b=S_ovlp, lower=True)

        return v[0] + self.c0 + self.nuclear_repulsion

    def one_ct1_rdm(self, fragment):
        r"""
        .. math::
            \langle a_{l\sigma}^\dagger E_{pq} a_{u\sigma} \rangle
        """
        if fragment == 'l1':
            return .5 * (self.two_rdm_l1_ord - np.einsum(
                'pu, lq->lupq', np.eye(self.ncas_l1), self.one_rdm_l1))
        elif fragment == 'l2':
            return .5 * (self.two_rdm_l2_ord - np.einsum(
                'pu, lq->lupq', np.eye(self.ncas_l2), self.one_rdm_l2))

    def one_ct2_rdm(self, fragment):
        r"""
        .. math::
            \langle a_{k\sigma} E_{pq} a_{t\sigma}^\dagger \rangle
        """
        if fragment == 'l1':
            return -.5 * self.two_rdm_l1_ord + np.einsum(
                'kt, pq->tkpq', np.eye(self.ncas_l1), self.one_rdm_l1) - .5 * np.einsum(
                    'qt, pk->tkpq', np.eye(self.ncas_l1), self.one_rdm_l1) + np.einsum(
                        'pk, qt->tkpq', np.eye(self.ncas_l1), np.eye(self.ncas_l1))
        elif fragment == 'l2':
            return -.5 * self.two_rdm_l2_ord + np.einsum(
                'kt, pq->tkpq', np.eye(self.ncas_l2), self.one_rdm_l2) - .5 * np.einsum(
                    'qt, pk->tkpq', np.eye(self.ncas_l2), self.one_rdm_l2) + np.einsum(
                        'pk, qt->tkpq', np.eye(self.ncas_l2), np.eye(self.ncas_l2))

    def two_ct1_rdm(self, fragment):
        r"""
        .. math::
            \langle a_{l\sigma}^\dagger E_{pq} E_{rs} a_{u\sigma} \rangle
        """
        if fragment == 'l1':
            return .5 * self.three_rdm_l1 - .5 * np.einsum(
                'pu, lqrs->lupqrs', np.eye(self.ncas_l1), self.two_rdm_l1_ord) - .5 * np.einsum(
                    'ru, lspq->lupqrs', np.eye(self.ncas_l1), self.two_rdm_l1_ord) + .5 * np.einsum(
                        'ru, ps, lq->lupqrs', np.eye(
                            self.ncas_l1), np.eye(self.ncas_l1),
                        self.one_rdm_l1)
        elif fragment == 'l2':
            return .5 * self.three_rdm_l2 - .5 * np.einsum(
                'pu, lqrs->lupqrs', np.eye(self.ncas_l2), self.two_rdm_l2_ord) - .5 * np.einsum(
                    'ru, lspq->lupqrs', np.eye(self.ncas_l2), self.two_rdm_l2_ord) + .5 * np.einsum(
                        'ru, ps, lq->lupqrs', np.eye(
                            self.ncas_l2), np.eye(self.ncas_l2),
                        self.one_rdm_l2)

    def two_ct2_rdm(self, fragment):
        r"""
        .. math::
            \langle a_{k\sigma} E_{pq} E_{rs} a_{t\sigma}^\dagger \rangle
        """
        if fragment == 'l1':
            return sum((
                - .5 * self.three_rdm_l1,
                np.einsum('tk, pqrs->tkpqrs', np.eye(self.ncas_l1),
                          self.two_rdm_l1_ord),
                - .5 * np.einsum('tq, pkrs->tkpqrs',
                                 np.eye(self.ncas_l1), self.two_rdm_l1_ord),
                np.einsum('tq, kp, rs->tkpqrs', np.eye(self.ncas_l1), np.eye(self.ncas_l1),
                          self.one_rdm_l1),
                - .5 * np.einsum('ts, rkpq->tkpqrs',
                                 np.eye(self.ncas_l1), self.two_rdm_l1_ord),
                np.einsum('ts, kr, pq->tkpqrs', np.eye(self.ncas_l1), np.eye(self.ncas_l1),
                          self.one_rdm_l1),
                - .5 * np.einsum('ts, qr, pk->tkpqrs', np.eye(self.ncas_l1), np.eye(self.ncas_l1),
                                 self.one_rdm_l1),
                np.einsum('ts, pk, qr->tkpqrs', np.eye(self.ncas_l1), np.eye(self.ncas_l1),
                          np.eye(self.ncas_l1))
            ))
        elif fragment == 'l2':
            return sum((
                - .5 * self.three_rdm_l2,
                np.einsum('tk, pqrs->tkpqrs', np.eye(self.ncas_l2),
                          self.two_rdm_l2_ord),
                - .5 * np.einsum('tq, pkrs->tkpqrs',
                                 np.eye(self.ncas_l2), self.two_rdm_l2_ord),
                np.einsum('tq, kp, rs->tkpqrs', np.eye(self.ncas_l2), np.eye(self.ncas_l2),
                          self.one_rdm_l2),
                - .5 * np.einsum('ts, rkpq->tkpqrs',
                                 np.eye(self.ncas_l2), self.two_rdm_l2_ord),
                np.einsum('ts, rk, pq->tkpqrs', np.eye(self.ncas_l2), np.eye(self.ncas_l2),
                          self.one_rdm_l2),
                - .5 * np.einsum('ts, qr, pk->tkpqrs', np.eye(self.ncas_l2), np.eye(self.ncas_l2),
                                 self.one_rdm_l2),
                np.einsum('ts, pk, qr->tkpqrs', np.eye(self.ncas_l2), np.eye(self.ncas_l2),
                          np.eye(self.ncas_l2)),
            ))

    def zero_ct2_rdm(self, fragment):
        r"""
        .. math::
            \langle a_{t\sigma} a_{k\sigma}^\dagger \rangle
        """
        if fragment == 'l1':
            return - .5 * self.one_rdm_l1 + np.eye(self.ncas_l1)
        elif fragment == 'l2':
            return - .5 * self.one_rdm_l2 + np.eye(self.ncas_l2)

    def get_ct_frag_rdms(self, ct):
        if ct == 'A':
            zero_ct_rdm_l1 = self.zero_ct2_rdm('l1')
            one_ct_rdm_l1 = self.one_ct2_rdm('l1')
            two_ct_rdm_l1 = self.two_ct2_rdm('l1')
            zero_ct_rdm_l2 = .5 * self.one_rdm_l2
            one_ct_rdm_l2 = self.one_ct1_rdm('l2')
            two_ct_rdm_l2 = self.two_ct1_rdm('l2')
        elif ct == 'B':
            zero_ct_rdm_l1 = .5 * self.one_rdm_l1
            one_ct_rdm_l1 = self.one_ct1_rdm('l1')
            two_ct_rdm_l1 = self.two_ct1_rdm('l1')
            zero_ct_rdm_l2 = self.zero_ct2_rdm('l2')
            one_ct_rdm_l2 = self.one_ct2_rdm('l2')
            two_ct_rdm_l2 = self.two_ct2_rdm('l2')

        return (zero_ct_rdm_l1, one_ct_rdm_l1, two_ct_rdm_l1,
                zero_ct_rdm_l2, one_ct_rdm_l2, two_ct_rdm_l2)

    def get_h0_pure_ct(self, fragment, ct):
        (zero_ct_rdm_l1, one_ct_rdm_l1, two_ct_rdm_l1,
         zero_ct_rdm_l2, one_ct_rdm_l2, two_ct_rdm_l2) = self.get_ct_frag_rdms(ct)

        if fragment == 'l1':
            c1_pure = self.c1[np.ix_(*[self.l1_idx]*2)]
            c2_pure = self.c2[np.ix_(*[self.l1_idx]*4)]
            c1 = h_prime(c1_pure, c2_pure)

            if ct == 'A':
                h1 = 2 * np.einsum('pq, tkpq, lu->kltu', c1,
                                   one_ct_rdm_l1, zero_ct_rdm_l2)
                h2 = np.einsum('pqrs, tkpqrs, lu->kltu',
                               c2_pure, two_ct_rdm_l1, zero_ct_rdm_l2)
            elif ct == 'B':
                h1 = 2 * np.einsum('pq, lupq, tk->kltu', c1,
                                   one_ct_rdm_l1, zero_ct_rdm_l2)
                h2 = np.einsum('pqrs, lupqrs, tk->kltu',
                               c2_pure, two_ct_rdm_l1, zero_ct_rdm_l2)
        elif fragment == 'l2':
            c1_pure = self.c1[np.ix_(*[self.l2_idx]*2)]
            c2_pure = self.c2[np.ix_(*[self.l2_idx]*4)]
            c1 = h_prime(c1_pure, c2_pure)

            if ct == 'A':
                h1 = 2 * np.einsum('pq, lupq, tk->kltu', c1,
                                   one_ct_rdm_l2, zero_ct_rdm_l1)
                h2 = np.einsum('pqrs, lupqrs, tk->kltu',
                               c2_pure, two_ct_rdm_l2, zero_ct_rdm_l1)
            elif ct == 'B':
                h1 = 2 * np.einsum('pq, tkpq, lu->kltu', c1,
                                   one_ct_rdm_l2, zero_ct_rdm_l1)
                h2 = np.einsum('pqrs, tkpqrs, lu->kltu',
                               c2_pure, two_ct_rdm_l2, zero_ct_rdm_l1)
        return h1 + h2

    def get_h0_disp_ct(self, ct):
        (zero_ct_rdm_l1, one_ct_rdm_l1, two_ct_rdm_l1,
         zero_ct_rdm_l2, one_ct_rdm_l2, two_ct_rdm_l2) = self.get_ct_frag_rdms(ct)

        if ct == 'A':
            h1 = 2 * np.einsum('pqrs, rs, tkpq, lu->kltu', self.exchange,
                               self.one_rdm_l2, one_ct_rdm_l1, zero_ct_rdm_l2)
            h2 = 2 * np.einsum('pqrs, pq, tk, lurs->kltu', self.exchange,
                               self.one_rdm_l1, zero_ct_rdm_l1, one_ct_rdm_l2)
        if ct == 'B':
            h1 = 2 * np.einsum('pqrs, rs, lupq, tk->kltu', self.exchange,
                               self.one_rdm_l2, one_ct_rdm_l1, zero_ct_rdm_l2)
            h2 = 2 * np.einsum('pqrs, pq, lu, tkrs->kltu', self.exchange,
                               self.one_rdm_l1, zero_ct_rdm_l1, one_ct_rdm_l2)

        h3 = - np.einsum('pqrs, pq, rs', self.exchange,
                         self.one_rdm_l1, self.one_rdm_l2) * self.get_overlap_ct(ct)
        return h1 + h2 + h3

    def get_overlap_ct(self, ct):
        if ct == 'A':
            return np.einsum('kt, lu->kltu', self.zero_ct2_rdm('l1'), self.one_rdm_l2)
        elif ct == 'B':
            return np.einsum('lu, kt->kltu', self.one_rdm_l1, self.zero_ct2_rdm('l2'))

    def get_ham_ct_col(self, ct):
        (zero_ct_rdm_l1, _, _,
         zero_ct_rdm_l2, _, _) = self.get_ct_frag_rdms(ct)
        g_eff_A = self.c2[np.ix_(
            self.l1_idx, self.l1_idx, self.l1_idx, self.l2_idx)]
        g_eff_B = self.c2[np.ix_(
            self.l2_idx, self.l2_idx, self.l2_idx, self.l1_idx)]
        if ct == 'A':
            one_rdm_ct_l1 = - .5 * self.two_rdm_l1_ord + np.einsum(
                'pq, tr->pqtr', self.one_rdm_l1, np.eye(self.ncas_l1))
            h_eff = self.c1[np.ix_(self.l2_idx, self.l1_idx)] - np.einsum(
                'prrq->pq', self.c2[np.ix_(self.l2_idx, self.l2_idx, self.l2_idx, self.l1_idx)])

            h1 = np.einsum('pq, tq, pu->tu', h_eff,
                           zero_ct_rdm_l1, self.one_rdm_l2)
            h2_A = np.einsum('pqrs, pqtr, su->tu', g_eff_A,
                             one_rdm_ct_l1, self.one_rdm_l2)
            h2_B = np.einsum('pqrs, ts, pqru->tu', g_eff_B,
                             zero_ct_rdm_l1, self.two_rdm_l2_ord)
        elif ct == 'B':
            one_rdm_ct_l2 = - .5 * self.two_rdm_l2_ord + np.einsum(
                'pq, tr->pqtr', self.one_rdm_l2, np.eye(self.ncas_l2))
            h_eff = self.c1[np.ix_(self.l1_idx, self.l2_idx)] - np.einsum(
                'prrq->pq', self.c2[np.ix_(self.l1_idx, self.l1_idx, self.l1_idx, self.l2_idx)])

            h1 = np.einsum('pq, pu, tq->tu', h_eff,
                           self.one_rdm_l1, zero_ct_rdm_l2)
            h2_A = np.einsum('pqrs, pqru, ts->tu', g_eff_A,
                             self.two_rdm_l1_ord, zero_ct_rdm_l2)
            h2_B = np.einsum('pqrs, su, pqtr->tu', g_eff_B,
                             self.one_rdm_l1, one_rdm_ct_l2)
        return h1 + h2_A + h2_B

    def run_fragpt2_ct(self):
        self.get_four_rdms()

        nex_A = self.ncas_l1 * self.ncas_l2
        E_0 = self.h0_expval(w_core=False)
        S_ovlp_A = self.get_overlap_ct('A').reshape((nex_A, nex_A))
        H_l1_A = self.get_h0_pure_ct('l1', 'A')
        H_l2_A = self.get_h0_pure_ct('l2', 'A')
        H_0_ct_A = self.get_h0_disp_ct('A')
        H_0_A = H_l1_A + H_l2_A + H_0_ct_A
        H_0_A = H_0_A.reshape((nex_A, nex_A))

        print("Hermitian check A, should be true:")
        print(np.allclose(H_l1_A, np.einsum('kltu->tukl', H_l1_A)))
        print(np.allclose(H_l2_A, np.einsum('kltu->tukl', H_l2_A)))
        print(np.allclose(H_0_ct_A, np.einsum('kltu->tukl', H_0_ct_A)))
        print(np.allclose(H_0_A, H_0_A.T))

        H_prime_col_A = self.get_ham_ct_col('A')
        H_prime_col_A = H_prime_col_A.reshape((nex_A))

        self.psi1_c_ct_A = np.linalg.solve((H_0_A - E_0 * S_ovlp_A),
                                           -H_prime_col_A)

        e_pt2_A = np.einsum('n, n', H_prime_col_A,
                            self.psi1_c_ct_A)

        nex_B = self.ncas_l2 * self.ncas_l1
        S_ovlp_B = self.get_overlap_ct('B').reshape((nex_B, nex_B))
        H_l1_B = self.get_h0_pure_ct('l1', 'B')
        H_l2_B = self.get_h0_pure_ct('l2', 'B')
        H_0_ct_B = self.get_h0_disp_ct('B')
        H_0_B = H_l1_B + H_l2_B + H_0_ct_B
        H_0_B = H_0_B.reshape((nex_B, nex_B))

        H_prime_col_B = self.get_ham_ct_col('B')
        H_prime_col_B = H_prime_col_B.reshape((nex_B))

        print("Hermitian check B, should be true:")
        print(np.allclose(H_l1_B, np.einsum('kltu->tukl', H_l1_B)))
        print(np.allclose(H_l2_B, np.einsum('kltu->tukl', H_l2_B)))
        print(np.allclose(H_0_ct_B, np.einsum('kltu->tukl', H_0_ct_B)))
        print(np.allclose(H_0_B, H_0_B.T))

        self.psi1_c_ct_B = np.linalg.solve((H_0_B - E_0 * S_ovlp_B),
                                           -H_prime_col_B)

        e_pt2_B = np.einsum('n, n', H_prime_col_B,
                            self.psi1_c_ct_B)

        return e_pt2_A, e_pt2_B


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

    # # # for CAS(4,4)(4,4):
    # act_idx_l1_base1 = [4, 5, 11, 12]
    # act_idx_l2_base1 = [8, 9, 27, 28]

    # # for CAS(2,2)(4,4):
    act_idx_l1_base1 = [5, 11]
    act_idx_l2_base1 = [8, 9, 27, 28]

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

    # # SET HERE THE SIZE OF THE ACTIVE SPACE
    # ncas = len(act_idx)
    # nelecas = ncas - 2 * (ncas // 2 % 2)
    # nelec_l1 = nelecas // 2
    # nelec_l2 = nelecas // 2
    # ncas_l1 = ncas // 2
    # ncas_l2 = ncas // 2

    ncas = len(act_idx)
    # nelecas = 6
    nelec_l1 = 2
    nelec_l2 = 4
    ncas_l1 = 2
    ncas_l2 = 4

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
    # casci.run_ct1_casci(nroots)
    # casci.run_ct2_casci(nroots)
    e_tot, e_l1, e_l2 = casci.run_self_consistent_casci()

    e_corr_naive = casci.e_l1 + casci.e_l2 - 2 * mf.e_tot
    e_sumcorr = casci.e_l1 + casci.e_l2 - mf.e_tot

    print(f'Naive energy:            {casci.e_naive:.6f}')
    print(f'Sum of corr energies:    {e_sumcorr:.6f}')
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
