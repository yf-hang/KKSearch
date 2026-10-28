"""Numerical checks and post-search interpretation of frozen candidates."""
from fractions import Fraction
from itertools import combinations

import numpy as np
from physics import SECTORS, STATES, matrix
from utils import SPLITS

def relative_residual(matrix, coefficients):
    x = np.asarray(matrix, dtype=np.float64)
    c = np.atleast_2d(np.asarray(coefficients, dtype=np.float64))
    if x.ndim != 2 or c.shape[1] != x.shape[1]:
        raise ValueError("Matrix and coefficient dimensions do not agree")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(c)):
        raise ValueError("Non-finite data or coefficients")
    if np.any(np.count_nonzero(c, axis=1) < 2):
        # Disallow a single term and 0=0. Callers handle these as invalid candidates.
        raise ValueError("An identity needs at least two nonzero terms")
    numerator = x @ c.T
    scale = np.abs(x) @ np.abs(c).T
    # Absolute floor prevents flat-space selection-rule zeros from amplifying
    # quadrature roundoff into spurious relative violations.
    floor = max(float(np.max(np.abs(x))) * 1e-10, 1e-30)
    active = scale > floor
    rr = numerator / np.maximum(scale, floor)
    # A candidate zero on most rows does not provide useful evidence.
    coverage = np.mean(active, axis=0)
    rms = np.sqrt(np.sum(rr**2 * active, axis=0) / np.maximum(np.sum(active, axis=0), 1))
    maximum = np.max(np.abs(rr) * active, axis=0)
    return rms, maximum, coverage


def evidence(x, c, tolerance=1e-8):
    rms, maximum, coverage = relative_residual(x, c)
    return {"rms_relative":float(rms[0]),"max_relative":float(maximum[0]),
            "active_fraction":float(coverage[0]),"tolerance":tolerance,
            "passed":bool(maximum[0] < tolerance and coverage[0] >= .8)}



def checks(dataset,candidate,terms,independent=True):
    ctx=candidate['context'];c=candidate['coefficients'];out={}
    with np.load(dataset) as data:
        for split in SPLITS[1:]:
            out[split]=evidence(data[split+'_matrix'][ctx],c)
        meta=data['test_modes_meta'][ctx]
    if independent:
        parts=[];zero=[]
        for q,b in np.unique(meta[:,:2],axis=0):
            rows=meta[(meta[:,0]==q)&(meta[:,1]==b),2:].astype(int)
            parts.append(matrix(q,b,rows,terms,[ctx],640,2e-12,'independent')[0])
            for slot,state in enumerate(SECTORS[ctx]):
                if STATES[state]['zero_mode']:
                    zr=rows[:8].copy();zr[:,slot]=0
                    zero.append(matrix(q,b,zr,terms,[ctx],640,2e-12,'independent')[0])
        out['independent']=evidence(np.concatenate(parts),c)
        if zero:
            try:
                z=np.concatenate(zero); score=evidence(z,c)
                denominator=np.abs(z)@np.abs(c)
                residual=np.abs(z@c)
                floor=max(np.abs(z).max()*1e-10,1e-30)
                # Zero-mode identities can become 0=0; count these separately.
                score['degenerate_rows']=int(np.sum(denominator<=floor))
                score['passed']=bool(np.all(residual<=1e-8*np.maximum(denominator,floor)))
                out['zero_modes']=score
            except ValueError: out['zero_modes']={'passed':False}
    return out




def _key(weight, derivatives=(), factors=()):
    return weight, tuple(sorted(derivatives)), tuple(sorted(factors))


def projection_rows(terms, states):
    """Published equations + product-rule expansions for post-search explanation."""
    index = {_key(t['weight'], t['derivatives'], t['factors']): i
             for i, t in enumerate(terms)}
    rows = []
    def add(label, entries):
        if not all(key in index for key, _ in entries):
            return
        row = [0] * len(terms)
        for key, value in entries:
            row[index[key]] += value
        rows.append((label, row))
    for slot, state in enumerate(states):
        entries = [(_key(3, factors=(f'M{slot}', f'M{slot}')), 1)]
        entries += [(_key(3, (slot, j)), -1) for j in range(len(states)) if j != slot]
        if state == 'u':
            add(f'2406.12713v3 (D.7), external slot {slot}', entries)
        elif state == 's':
            add(f'2504.05199v3 (3.6), elementary scalar projection, slot {slot}',
                entries + [(_key(5, factors=('b', 'b')), -1)])
        elif state == 'v' and states.count('v') >= 2:
            add(f'2504.05199v3 (2.18b), elementary vector projection, slot {slot}',
                entries + [(_key(5, factors=('q', 'q')), 3)])
        elif state == 'w' and 'v' in states:
            add(f'2504.05199v3 (2.23), elementary scalar-Goldstone projection, slot {slot}',
                entries + [(_key(5, factors=('q', 'q')), 4)])
    # Same-family potentials and endpoint terms cancel in the Wronskian.
    for i, j in combinations(range(len(states)), 2):
        if states[i] != states[j]:
            continue
        entries = [(_key(3, factors=(f'M{i}', f'M{i}')), 1),
                   (_key(3, factors=(f'M{j}', f'M{j}')), -1)]
        for k in range(len(states)):
            if k not in (i, j):
                entries += [(_key(3, (i, k)), -1), (_key(3, (j, k)), 1)]
        add(f'same-family Wronskian from published mode equations, slots {i},{j}', entries)
    return rows


def review(terms, states, coefficients):
    """Exact rational span membership; called only on already frozen output."""
    basis = {}
    labels = []
    for label, row in projection_rows(terms, states):
        v = list(map(Fraction, row))
        for pivot, b in sorted(basis.items()):
            if v[pivot]:
                value = v[pivot]
                v = [x-value*y for x, y in zip(v, b)]
        pivot = next((i for i, x in enumerate(v) if x), None)
        if pivot is not None:
            scale = v[pivot]
            basis[pivot] = [x/scale for x in v]
            labels.append(label)
    v = list(map(Fraction, coefficients))
    if not any(v):
        return {'status': 'rejected_zero_vector', 'accepted_as_new': False}
    for pivot, b in sorted(basis.items()):
        value = v[pivot]
        v = [x-value*y for x, y in zip(v, b)]
    derived = not any(v)
    return {'status': 'derived_from_published_premises' if derived else 'requires_literature_review',
            'accepted_as_new': None, 'rejected_as_known': False,
            'explicitly_stated_in_references': None,
            'reason': 'Exact linear combination of projections of published mode equations. This establishes derivability, not that the explicit six-point conclusion appears in either paper.' if derived else 'Outside this limited projection span. Explicit occurrence still requires a paper comparison.',
            'premise_basis': labels, 'basis_rank': len(basis),
            'learner_received_review': False}
