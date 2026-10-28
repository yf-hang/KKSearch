"""Warped background, KK profiles, six-point dictionary and numerical data."""
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, combinations_with_replacement

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_bvp, solve_ivp
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq
from scipy.special import roots_legendre, jv, yv
from scipy.stats import qmc
from utils import SPLITS, sha256_file

@dataclass(frozen=True)
class Background:
    q: float
    bulk: float

    def __post_init__(self):
        if not (np.isfinite(self.q) and np.isfinite(self.bulk)):
            raise ValueError("Background parameters must be finite")
        if self.q < 0 or self.bulk < 0:
            raise ValueError("This implementation requires q>=0 and bulk>=0")


@dataclass
class ModeSet:
    eigenvalues: np.ndarray
    x: np.ndarray
    dx: np.ndarray
    values: np.ndarray
    derivatives: np.ndarray
    gram_error: float
    bc_error: float
    bvp_residual: float
    max_nodes: int


@lru_cache(maxsize=16)
def quadrature(order):
    if order < 32:
        raise ValueError("quadrature order must be at least 32")
    x, dx = roots_legendre(order)
    return (x + 1) / 2, dx / 2


def finite_volume(q, bulk, n_modes=5, elements=256):
    """Second-order, symmetric conservative discretization, including mode 0."""
    Background(q, bulk)
    if elements < max(32, 8 * n_modes):
        raise ValueError("Insufficient finite-volume resolution")
    x = np.linspace(0, 1, elements + 1)
    h = 1 / elements
    w = (1 + q * x) ** -3
    edge = (1 + q * (x[:-1] + h / 2)) ** -3 / h
    mass = w * h
    mass[[0, -1]] *= 0.5
    potential = bulk**2 * (1 + q * x) ** -2
    diagonal = np.r_[edge[0], edge[:-1] + edge[1:], edge[-1]] / mass + potential
    off = -edge / np.sqrt(mass[:-1] * mass[1:])
    vals, vecs = eigh_tridiagonal(diagonal, off, select="i",
                                select_range=(0, n_modes - 1), tol=1e-11)
    f = (vecs / np.sqrt(mass[:, None])).T
    f *= np.where(f[:, :1] >= 0, 1.0, -1.0)
    return x, np.maximum(vals, 0), f


@lru_cache(maxsize=64)
def solve_modes(q, bulk, n_modes=5, order=192, tol=2e-9):
    """Solve normalized eigenmodes; throws on convergence or mode-order failure."""
    q, bulk = float(q), float(bulk)
    Background(q, bulk)
    if n_modes < 1 or tol <= 0:
        raise ValueError("n_modes and tol must be positive")
    x, dx = quadrature(order)
    mesh, guesses, functions = finite_volume(q, bulk, n_modes)
    w = (1 + q * x) ** -3
    all_f, all_d, vals, residuals, bcs, nodes = [], [], [], [], [], []
    for j in range(n_modes):
        if bulk == 0 and j == 0:
            # Known constant zero mode; no division by its mass.
            f = np.full_like(x, 1 / np.sqrt(np.sum(dx * w)))
            all_f.append(f); all_d.append(np.zeros_like(x)); vals.append(0.0)
            residuals.append(0.0); bcs.append(0.0); nodes.append(0)
            continue
        f0 = functions[j]
        wm = (1 + q * mesh) ** -3
        v0 = wm * np.gradient(f0, mesh, edge_order=2)
        v0[[0, -1]] = 0
        norm0 = cumulative_trapezoid(wm * f0**2, mesh, initial=0)
        norm = np.sqrt(norm0[-1])
        initial = np.stack([f0 / norm, v0 / norm, norm0 / norm**2])

        def ode(t, y, lam):
            wt = (1 + q * t) ** -3
            vt = bulk**2 * (1 + q * t) ** -5
            return np.array([y[1] / wt, (vt - lam[0] * wt) * y[0], wt * y[0]**2])

        def boundary(ya, yb, lam):
            return np.array([ya[1], yb[1], ya[2], yb[2] - 1])

        sol = solve_bvp(ode, boundary, mesh, initial, p=[guesses[j]],
                        tol=tol, bc_tol=tol * 0.1, max_nodes=30000)
        if not sol.success:
            raise RuntimeError(f"BVP failed for q={q}, bulk={bulk}, mode={j}: {sol.message}")
        yy = sol.sol(x)
        phase = 1 if sol.sol(0)[0] >= 0 else -1
        norm = np.sqrt(np.sum(dx * w * yy[0]**2))
        all_f.append(phase * yy[0] / norm)
        all_d.append(phase * yy[1] / w / norm)
        vals.append(float(sol.p[0]))
        residuals.append(float(np.max(sol.rms_residuals)))
        bcs.append(float(np.max(np.abs(boundary(sol.sol(0), sol.sol(1), sol.p)))))
        nodes.append(int(sol.x.size))
    f, d, vals = np.array(all_f), np.array(all_d), np.array(vals)
    if np.any(np.diff(vals) <= 0) or np.any(vals < -1e-10):
        raise RuntimeError("Unordered, duplicated, or negative eigenvalues")
    # Oscillation theorem detects converging to the wrong branch.
    for j, fj in enumerate(f):
        signs = np.sign(fj[np.abs(fj) > 1e-8 * np.max(np.abs(fj))])
        if np.sum(signs[:-1] * signs[1:] < 0) != j:
            raise RuntimeError(f"Unexpected node count for mode {j}")
    gram_error = float(np.max(np.abs((f * (dx * w)) @ f.T - np.eye(n_modes))))
    if gram_error > 1e-6:
        raise RuntimeError(f"Mode orthogonality failed: {gram_error}")
    return ModeSet(vals, x, dx, f, d, gram_error, max(bcs), max(residuals), max(nodes))


@lru_cache(maxsize=32)
def shooting_modes(q, bulk, n_modes=5, order=384, tol=2e-12):
    """Independent DOP853 shooting check; no BVP solution or identity supplied.

    Finite-volume eigenvalues give local root brackets. Each root is refined by
    integrating the initial-value problem f(0)=1, w*f'(0)=0 and imposing f'(1)=0.
    Intended for the low-lying modes and parameter ranges in this project.
    """
    Background(q, bulk)
    if n_modes < 1 or tol <= 0:
        raise ValueError("n_modes and tol must be positive")
    x, dx = quadrature(order)
    w = (1 + q*x)**-3
    _, guesses, _ = finite_volume(q, bulk, n_modes+1, 512)
    functions, derivatives, eigenvalues, boundary_errors = [], [], [], []
    for j in range(n_modes):
        if bulk == 0 and j == 0:
            functions.append(np.ones_like(x)/np.sqrt(np.sum(dx*w)))
            derivatives.append(np.zeros_like(x)); eigenvalues.append(0.0); boundary_errors.append(0.0)
            continue

        def integrate(lam, dense=False):
            def ode(t, y):
                wt = (1+q*t)**-3
                potential = bulk**2*(1+q*t)**-5
                return [y[1]/wt, (potential-lam*wt)*y[0]]
            sol = solve_ivp(ode, (0., 1.), [1., 0.], method="DOP853", rtol=tol,
                            atol=tol*.05, dense_output=dense)
            if not sol.success:
                raise RuntimeError(f"Shooting IVP failed: {sol.message}")
            return sol

        gap = guesses[j+1]-guesses[j]
        if j:
            gap = min(gap, guesses[j]-guesses[j-1])
        lower = max(0.0, guesses[j]-.2*gap)
        upper = guesses[j]+.2*gap
        eigenvalue = brentq(lambda lam: integrate(lam).y[1, -1], lower, upper,
                            xtol=5e-13, rtol=5e-15)
        solution = integrate(eigenvalue, dense=True)
        values = solution.sol(x)
        norm = np.sqrt(np.sum(dx*w*values[0]**2))
        f, derivative = values[0]/norm, values[1]/(w*norm)
        signs = np.sign(f[np.abs(f) > np.max(np.abs(f))*1e-8])
        if np.sum(signs[:-1]*signs[1:] < 0) != j:
            raise RuntimeError("Shooting root converged to the wrong mode")
        functions.append(f); derivatives.append(derivative); eigenvalues.append(eigenvalue)
        boundary_errors.append(abs(solution.y[1,-1]/norm))
    f = np.array(functions)
    gram = float(np.max(np.abs((f*(dx*w))@f.T-np.eye(n_modes))))
    if gram > 1e-7:
        raise RuntimeError(f"Shooting orthogonality check failed: {gram}")
    return ModeSet(np.array(eigenvalues), x, dx, f, np.array(derivatives), gram,
                   max(boundary_errors), 0.0, 0)


# Field profiles and independent Bessel calculation.

STATES = {
    's': {'field': 'scalar matter phi, phi*', 'helicities': [0], 'zero_mode': True},
    'u': {'field': 'graviton h_mu_nu', 'helicities': [-2,-1,0,1,2], 'zero_mode': True},
    'v': {'field': 'gravitational vector Goldstone V_mu', 'helicities': [-1,0,1], 'zero_mode': False},
    'w': {'field': 'gravitational scalar Goldstone; radion at n=0', 'helicities': [0], 'zero_mode': True},
}

@lru_cache(maxsize=64)
def _bessel_tower(q, n_modes, order):
    x, dx = quadrature(order)
    t = 1+q*x
    nu = 2
    boundary_order = nu-1
    def determinant(m):
        a,b=m/q,m*(1+q)/q
        return jv(boundary_order,a)*yv(boundary_order,b)-yv(boundary_order,a)*jv(boundary_order,b)
    roots=[]
    left=1e-5
    while len(roots)<n_modes-1:
        right=left+np.pi/12
        if determinant(left)*determinant(right)<0:
            roots.append(brentq(determinant,left,right,xtol=2e-13))
        left=right
        if left>1000:
            raise RuntimeError('Bessel root scan failed')
    values=[np.ones_like(x)/np.sqrt(np.sum(dx*t**-3))]
    deriv=[np.zeros_like(x)]
    partners=[]
    for mass in roots:
        a=mass/q
        def raw(v):
            return t**nu*(yv(boundary_order,a)*jv(v,mass*t/q)-jv(boundary_order,a)*yv(v,mass*t/q))
        f=raw(nu)
        at0=yv(boundary_order,a)*jv(nu,a)-jv(boundary_order,a)*yv(nu,a)
        norm=np.sqrt(np.sum(dx*t**-3*f*f))*np.sign(at0)
        values.append(f/norm)
        deriv.append(mass*raw(nu-1)/norm)
        partners.append((-raw(nu-1)/norm, -raw(nu-2)/norm))
    return x,dx,np.r_[0,np.square(roots)],np.array(values),np.array(deriv),partners

@lru_cache(maxsize=64)
def boson_modes(q, bulk, n_modes, order=320, tol=2e-9, method='bvp'):
    if q <= 0:
        raise ValueError('Bosonic mode bank currently requires q>0')
    scalar=(shooting_modes if method=='independent' else solve_modes)(q,bulk,n_modes,order,tol)
    x,dx=scalar.x,scalar.dx
    t=1+q*x
    ap=-q/t
    bank={'s':(scalar.values,scalar.derivatives,scalar.eigenvalues)}
    if method=='independent':
        _,_,lam,f,d,other=_bessel_tower(q,n_modes,order)
        v=np.vstack([np.full_like(x,np.nan),*[pair[0] for pair in other]])
    else:
        modes=solve_modes(q,0.,n_modes,order,tol)
        lam,f,d=modes.eigenvalues,modes.values,modes.derivatives
        v=np.full_like(f,np.nan)
        v[1:]=-d[1:]/np.sqrt(lam[1:,None])
    dv=np.full_like(v,np.nan)
    dv[1:]=np.sqrt(lam[1:,None])*f[1:]-3*ap*v[1:]
    bank['u']=(f,d,lam)
    bank['v']=(v,dv,lam)
    w=np.empty_like(f)
    w[0]=t*t/np.sqrt(np.sum(dx*t))
    if method=='independent':
        w[1:]=np.array([pair[1] for pair in other])
    else:
        w[1:]=(dv[1:]+ap*v[1:])/np.sqrt(lam[1:,None])
    dw=-2*ap*w
    dw[1:]-=np.sqrt(lam[1:,None])*v[1:]
    bank['w']=(w,dw,lam)
    return x,dx,bank


# Six-point grammar, numerical integrals and data generation.

POINTS = 6
PROFILES = list(STATES)
SECTORS = list(combinations_with_replacement(PROFILES,POINTS))

def catalog(weights=(1,3,5,7),seed=620281):
    terms=[]
    factors=['q','b']+[f'M{i}' for i in range(POINTS)]
    for weight in weights:
        for degree in range(3):
            for slots in combinations(range(POINTS),degree):
                for masses in combinations_with_replacement(factors,2-degree):
                    terms.append({'weight':weight,'derivatives':list(slots),'factors':list(masses)})
    rng=np.random.default_rng(seed)
    return [terms[i] for i in rng.permutation(len(terms))]

def matrix(q,b,rows,terms,sector_ids=None,order=320,tol=2e-9,method='bvp'):
    rows=np.asarray(rows,dtype=int)
    if rows.ndim not in (2,3) or rows.shape[-1]!=POINTS or rows.size==0 or np.any(rows<0):
        raise ValueError('Expected nonnegative six-mode tuples')
    x,dx,bank=boson_modes(float(q),float(b),int(rows.max())+1,order,tol,method)
    sector_ids=range(len(SECTORS)) if sector_ids is None else sector_ids
    weights={t['weight']:dx*(1+q*x)**-t['weight'] for t in terms}
    patterns=sorted({tuple(t['derivatives']) for t in terms})
    output=[]
    for si in sector_ids:
        states=SECTORS[si]
        actual=rows[si] if rows.ndim==3 else rows
        for i,state in enumerate(states):
            if not STATES[state]['zero_mode'] and np.any(actual[:,i]==0):
                raise ValueError(f'{state} has no physical zero mode')
        f=np.stack([bank[state][0][actual[:,i]] for i,state in enumerate(states)],axis=1)
        d=np.stack([bank[state][1][actual[:,i]] for i,state in enumerate(states)],axis=1)
        factors={'q':np.full(len(actual),q),'b':np.full(len(actual),b)}
        factors.update({f'M{i}':np.sqrt(bank[state][2][actual[:,i]]) for i,state in enumerate(states)})
        integrals={}
        for slots in patterns:
            product=np.prod(np.stack([d[:,i] if i in slots else f[:,i] for i in range(POINTS)],axis=1),axis=1)
            for a,quad in weights.items():
                integrals[a,slots]=product@quad
        cols=[]
        for term in terms:
            value=integrals[term['weight'],tuple(term['derivatives'])].copy()
            for factor in term['factors']:
                value*=factors[factor]
            cols.append(value)
        output.append(np.array(cols).T)
    return np.array(output)

def generate(out,config,terms):
    if config.get('points')!=POINTS or config['train_mode_max']<POINTS:
        raise ValueError('Six-point generation needs six slots and at least six nonzero modes')
    out.mkdir(parents=True,exist_ok=True)
    arrays={}
    for index,split in enumerate(SPLITS):
        bounds=config['ood_ranges' if split=='test_ood' else 'train_ranges']
        unit=qmc.LatinHypercube(2,seed=config['seed']+index).random(config['backgrounds'][split])
        bg=qmc.scale(unit,[bounds[k][0] for k in ('q','bulk')],[bounds[k][1] for k in ('q','bulk')])
        rng=np.random.default_rng(config['seed']+100*index)
        blocks=[]; meta=[]
        high=config['test_mode_max'] if split=='test_modes' else config['train_mode_max']
        for number,(q,b) in enumerate(bg):
            all_rows=[]
            for states in SECTORS:
                tuples=set()
                while len(tuples)<config['tuples_per_background']:
                    row=tuple(map(int,rng.choice(np.arange(high+1),POINTS,replace=False)))
                    if any(row[i]==0 and not STATES[state]['zero_mode'] for i,state in enumerate(states)):
                        continue
                    if split=='test_modes' and max(row)<=config['train_mode_max']:
                        continue
                    tuples.add(row)
                all_rows.append(sorted(tuples))
            rows=np.array(all_rows)
            blocks.append(matrix(q,b,rows,terms,order=config['order'],tol=config['tol']))
            meta.append(np.array([[[q,b,*row] for row in group] for group in rows]))
            print(f'bosonic data {split} {number+1}/{len(bg)}: {len(SECTORS)} sectors',flush=True)
        arrays[split+'_matrix']=np.concatenate(blocks,axis=1)
        arrays[split+'_meta']=np.concatenate(meta,axis=1)
    np.savez_compressed(out/'data.npz',**arrays)
    return {'npz_sha256':sha256_file(out/'data.npz'),'rows':{s:arrays[s+'_meta'].shape[1] for s in SPLITS},
            'profiles':STATES,'sectors':[list(s) for s in SECTORS],'terms_per_sector':len(terms),
            'total_primitive_terms':len(SECTORS)*len(terms),'search_modes':'All allowed mode indices including tensor zero mode, radion and lowest matter mode; no vector Goldstone zero mode'}

def formula(terms,sector,c):
    states=SECTORS[sector]; pieces=[]
    for t,value in zip(terms,c):
        if not value: continue
        fields=' '.join(f"{'d_' if i in t['derivatives'] else ''}{state}{i+1}" for i,state in enumerate(states))
        masses='*'.join(t['factors'])
        pieces.append(f"{'+' if value>0 else '-'}{abs(value):g}*{masses+'*' if masses else ''}C{t['weight']}[{fields}]")
    return ' '.join(pieces)+' = 0'
