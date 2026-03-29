"""
CMAES-MAESTRO-ALMA: CMA-ES + Dynamic Scalarization Conductor + Open-Ended Memory Search
=========================================================================================
A three-layer self-restructuring optimization system combining NEW ideas from:
Source 1: CMA-ES (Hansen & Ostermeier, 2001) - Wikipedia metaheuristic family
  https://en.wikipedia.org/wiki/CMA-ES
Source 2: MAESTRO (Zhao et al., arXiv:2601.07208, Jan 2026)
  Dynamic scalarization conductor via contextual bandit bi-level optimization.
Source 3: ALMA (Xiong, Hu & Clune, arXiv:2602.07755, Feb 2026)
  Open-ended memory design search expressed as executable code.

Layer 1: CMA-ES optimizer that adapts covariance matrix to learn landscape
Layer 2: MAESTRO Conductor dynamically re-weights objectives via contextual bandit
Layer 3: ALMA memory design search + Differential Evolution cross-domain transfer
"""
import numpy as np
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from typing import List, Callable, Dict, Tuple, Optional
import hashlib, warnings
warnings.filterwarnings("ignore")
GLOBAL_SEED = 42
RNG = np.random.RandomState(GLOBAL_SEED)

def rk4_step(f, y, t, dt):
    k1 = dt * f(t, y)
    k2 = dt * f(t + dt/2, y + k1/2)
    k3 = dt * f(t + dt/2, y + k2/2)
    k4 = dt * f(t + dt, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

def rk4_integrate(f, y0, t_span, n_steps=100):
    t0, tf = t_span
    dt = (tf - t0) / n_steps
    y, t = y0.copy(), t0
    traj = [y.copy()]
    for _ in range(n_steps):
        y = rk4_step(f, y, t, dt)
        t += dt
        traj.append(y.copy())
    return np.array(traj)

def search_space_diameter(archive):
    if len(archive) < 2: return 0.0
    pts = np.array(archive)
    return float(np.max(cdist(pts, pts, metric="euclidean")))

def search_space_volume_proxy(archive):
    if len(archive) < 2: return 0.0
    return float(np.prod(np.std(np.array(archive), axis=0) + 1e-12))

class NoveltyDetector:
    def __init__(self, k_nearest=5, novelty_threshold=0.1):
        self.k_nearest = k_nearest
        self.novelty_threshold = novelty_threshold
        self.seen_hashes = set()
        self.archive = []
    def _hash(self, s):
        return hashlib.md5(np.round(s, 2).tobytes()).hexdigest()
    def novelty_score(self, c):
        if len(self.archive) < self.k_nearest: return float("inf")
        d = np.linalg.norm(np.array(self.archive) - c, axis=1)
        return float(np.mean(np.sort(d)[:min(self.k_nearest, len(d))]))
    def is_genuinely_novel(self, c):
        if self._hash(c) in self.seen_hashes: return False
        return self.novelty_score(c) > self.novelty_threshold
    def register(self, c):
        self.seen_hashes.add(self._hash(c))
        self.archive.append(c.copy())

def rastrigin(x):
    return float(10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x)))
def rosenbrock(x):
    return float(np.sum(100*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2))
def ackley(x):
    n = len(x)
    return float(-20*np.exp(-0.2*np.sqrt(np.sum(x**2)/n)) - np.exp(np.sum(np.cos(2*np.pi*x))/n) + 20 + np.e)

class CMAESLayer:
    """Layer 1: CMA-ES (Hansen & Ostermeier 2001). NEW vs 0327: replaces MAP-Elites
    with covariance matrix adaptation that learns landscape shape."""
    def __init__(self, objective, dim, bounds=(-5.12,5.12), sigma0=1.0, pop_size=None):
        self.objective, self.dim, self.bounds = objective, dim, bounds
        self.lam = pop_size or (4 + int(3*np.log(dim)))
        self.mu = self.lam // 2
        w = np.log(self.mu+0.5) - np.log(np.arange(1,self.mu+1))
        self.weights = w / np.sum(w)
        self.mueff = 1.0 / np.sum(self.weights**2)
        self.cc = (4+self.mueff/dim)/(dim+4+2*self.mueff/dim)
        self.cs = (self.mueff+2)/(dim+self.mueff+5)
        self.c1 = 2/((dim+1.3)**2+self.mueff)
        self.cmu = min(1-self.c1, 2*(self.mueff-2+1/self.mueff)/((dim+2)**2+self.mueff))
        self.damps = 1+2*max(0,np.sqrt((self.mueff-1)/(dim+1))-1)+self.cs
        self.chiN = np.sqrt(dim)*(1-1/(4*dim)+1/(21*dim**2))
        self.mean = RNG.uniform(bounds[0], bounds[1], dim)
        self.sigma = sigma0
        self.C, self.pc, self.ps = np.eye(dim), np.zeros(dim), np.zeros(dim)
        self.B, self.D, self.invsqrtC = np.eye(dim), np.ones(dim), np.eye(dim)
        self.history, self.all_solutions = [], []
        self.generation = 0
        self.best_ever_fitness, self.best_ever_x = float("inf"), None

    def _decompose_C(self):
        self.C = np.triu(self.C) + np.triu(self.C,1).T
        Dsq, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(Dsq, 1e-20))
        self.invsqrtC = self.B @ np.diag(1.0/self.D) @ self.B.T

    def step(self, n_evals=None):
        if self.generation % max(1,int(1/(self.c1+self.cmu)/self.dim/10)) == 0:
            self._decompose_C()
        sols, fits = [], []
        for _ in range(self.lam):
            x = np.clip(self.mean + self.sigma*(self.B@(self.D*RNG.randn(self.dim))),
                        self.bounds[0], self.bounds[1])
            sols.append(x); fits.append(self.objective(x))
            self.all_solutions.append(x.copy())
        idx = np.argsort(fits)
        sols = [sols[i] for i in idx]; fits = [fits[i] for i in idx]
        if fits[0] < self.best_ever_fitness:
            self.best_ever_fitness, self.best_ever_x = fits[0], sols[0].copy()
        old = self.mean.copy()
        self.mean = sum(self.weights[i]*sols[i] for i in range(self.mu))
        y = (self.mean - old) / self.sigma
        z = self.invsqrtC @ y
        self.ps = (1-self.cs)*self.ps + np.sqrt(self.cs*(2-self.cs)*self.mueff)*z
        hs = float((np.linalg.norm(self.ps)/np.sqrt(1-(1-self.cs)**(2*(self.generation+1)))/self.chiN) < 1.4+2/(self.dim+1))
        self.pc = (1-self.cc)*self.pc + hs*np.sqrt(self.cc*(2-self.cc)*self.mueff)*y
        art = np.array([(sols[i]-old)/self.sigma for i in range(self.mu)])
        self.C = ((1-self.c1-self.cmu+(1-hs)*self.c1*self.cc*(2-self.cc))*self.C
                  + self.c1*np.outer(self.pc,self.pc)
                  + self.cmu*(art.T@np.diag(self.weights)@art))
        self.sigma = max(self.sigma*np.exp((self.cs/self.damps)*(np.linalg.norm(self.ps)/self.chiN-1)), 1e-20)
        self.generation += 1; self.history.append(fits[0])
        return fits[0]

    def get_best(self):
        return (self.best_ever_x.copy(), self.best_ever_fitness) if self.best_ever_x is not None else (None, float("inf"))
    def get_all_positions(self):
        return [s.copy() for s in self.all_solutions[-self.lam*3:]]
    def get_covariance_condition(self):
        return float(np.max(self.D)/np.min(self.D)) if np.min(self.D)>0 else float("inf")

class MAESTROConductor:
    """Layer 2: Dynamic scalarization (MAESTRO arXiv:2601.07208). NEW vs 0327:
    replaces heuristic MetaRestructurer with contextual bandit Conductor."""
    def __init__(self, n_arms=4, context_dim=6, lr=0.1):
        self.n_arms, self.context_dim, self.lr = n_arms, context_dim, lr
        self.theta = RNG.randn(n_arms, context_dim) * 0.01
        self.reward_history, self.transformations_applied = [], []

    def _get_context(self, fh, dh, nc, cc, gen, asize):
        ctx = np.zeros(self.context_dim)
        if len(fh)>=2: ctx[0]=fh[-1]-fh[-2]; ctx[1]=np.std(fh[-5:]) if len(fh)>=5 else 1.0
        if len(dh)>=2: ctx[2]=dh[-1]-dh[-2]
        ctx[3]=np.log1p(nc); ctx[4]=np.log1p(cc) if cc<1e10 else 10.0; ctx[5]=gen/100.0
        return ctx

    def get_scalarization_weights(self, ctx):
        logits = self.theta @ ctx - np.max(self.theta @ ctx)
        w = np.exp(logits) / (np.sum(np.exp(logits))+1e-12)
        return 0.7*w + 0.3*RNG.dirichlet(np.ones(self.n_arms)*2.0)

    def build_composite_objective(self, base_obj, positions, weights):
        def loss(x):
            f = base_obj(x)
            nov = div = 0.0
            if len(positions)>2:
                d = np.linalg.norm(np.array(positions[-50:])-x, axis=1)
                nov = -np.mean(np.sort(d)[:min(5,len(d))])
                div = -np.linalg.norm(x - np.mean(positions[-20:], axis=0))
            dim = len(x)
            def gf(t,y):
                eps=1e-5; g=np.zeros(dim); f0=base_obj(y)
                for i in range(dim):
                    yp=y.copy(); yp[i]+=eps; g[i]=(base_obj(yp)-f0)/eps
                return -0.05*g
            sm = base_obj(rk4_integrate(gf, x, (0,0.5), 5)[-1])
            return float(np.dot(weights, [f, nov, div, sm]))
        return loss

    def update_conductor(self, ctx, w, reward):
        self.reward_history.append(reward)
        adv = reward - (np.mean(self.reward_history[-5:]) if len(self.reward_history)>=5 else 0)
        logits = self.theta @ ctx - np.max(self.theta @ ctx)
        p = np.exp(logits)/(np.sum(np.exp(logits))+1e-12)
        for a in range(self.n_arms):
            self.theta[a] += self.lr*(w[a]-p[a])*ctx*adv
        self.transformations_applied.append(f"cond_upd_adv={adv:.4f}")

@dataclass
class MemoryDesign:
    name: str
    retrieve_fn: Callable
    update_fn: Callable
    score: float = 0.0

class ALMAMemorySearch:
    """Layer 3: ALMA memory design search (arXiv:2602.07755). NEW vs 0327:
    evolving memory design library + Differential Evolution cross-domain."""
    def __init__(self, stag_win=5, exp_thr=0.01):
        self.diameter_history, self.volume_history = [], []
        self.stag_win, self.exp_thr = stag_win, exp_thr
        self.interventions, self.algo_families = [], set()
        self.memory_designs = self._init_designs()

    def _init_designs(self):
        def nn_ret(arc, q, k=5):
            if len(arc)<k: return arc[:]
            d=np.linalg.norm(np.array(arc)-q,axis=1); return [arc[i] for i in np.argsort(d)[:k]]
        def simple_upd(arc, s, mx=200):
            arc.append(s.copy())
            if len(arc)>mx: arc.pop(0)
            return arc
        def div_ret(arc, q, k=5):
            if len(arc)<k: return arc[:]
            pts=np.array(arc); sel=[0]
            for _ in range(k-1):
                bd,bi=-1,0
                for i in range(len(pts)):
                    if i in sel: continue
                    md=min(np.linalg.norm(pts[i]-pts[j]) for j in sel)
                    if md>bd: bd,bi=md,i
                sel.append(bi)
            return [arc[i] for i in sel]
        def rec_ret(arc, q, k=5):
            if len(arc)<k: return arc[:]
            n=len(arc); w=np.exp(np.linspace(-2,0,n))
            return [arc[i] for i in RNG.choice(n,min(k,n),replace=False,p=w/w.sum())]
        return [MemoryDesign("nn",nn_ret,simple_upd), MemoryDesign("diverse",div_ret,simple_upd),
                MemoryDesign("recency",rec_ret,simple_upd)]

    def synthesize_new_design(self):
        i,j = RNG.choice(len(self.memory_designs),2,replace=False)
        d1,d2 = self.memory_designs[i], self.memory_designs[j]
        name = f"hyb_{d1.name}_{d2.name}_{len(self.memory_designs)}"
        def hret(arc,q,k=5):
            h=max(1,k//2)
            r1=d1.retrieve_fn(arc,q,h); r2=d2.retrieve_fn(arc,q,k-h)
            return (r1+[s for s in r2 if not any(np.allclose(s,r) for r in r1)])[:k]
        nd = MemoryDesign(name, hret, d2.update_fn)
        self.memory_designs.append(nd)
        self.interventions.append(f"synth_{name}")
        return nd

    def record_metrics(self, pos):
        self.diameter_history.append(search_space_diameter(pos))
        self.volume_history.append(search_space_volume_proxy(pos))

    def is_expanding(self):
        if len(self.diameter_history)<self.stag_win: return True
        r=self.diameter_history[-self.stag_win:]
        return (r[-1]-r[0])/(abs(r[0])+1e-12) > self.exp_thr

    def is_rearranging(self):
        if len(self.volume_history)<self.stag_win: return False
        rv=self.volume_history[-self.stag_win:]; rd=self.diameter_history[-self.stag_win:]
        return (abs(rv[-1]-rv[0])/(abs(rv[0])+1e-12)<self.exp_thr and
                abs(rd[-1]-rd[0])/(abs(rd[0])+1e-12)<self.exp_thr)

    def inject_de(self, l1, npop=30, ngen=20, F=0.8, CR=0.9):
        """Differential Evolution cross-domain (NEW: not in 0327 which used SA/PSO)."""
        self.algo_families.add("differential_evolution"); self.interventions.append("DE_inject")
        dim,lo,hi = l1.dim, l1.bounds[0], l1.bounds[1]
        pop = RNG.uniform(lo,hi,(npop,dim))
        fit = np.array([l1.objective(p) for p in pop])
        de_sols = [pop[np.argmin(fit)].copy()]
        for _ in range(ngen):
            for i in range(npop):
                ids=[j for j in range(npop) if j!=i]
                a,b,c = pop[RNG.choice(ids,3,replace=False)]
                mut = np.clip(a+F*(b-c),lo,hi)
                cr = RNG.rand(dim)<CR
                if not np.any(cr): cr[RNG.randint(dim)]=True
                trial = np.where(cr,mut,pop[i])
                tf = l1.objective(trial)
                if tf<fit[i]: pop[i],fit[i]=trial,tf
            bi=np.argmin(fit); de_sols.append(pop[bi].copy())
            if fit[bi]<l1.best_ever_fitness:
                l1.best_ever_fitness,l1.best_ever_x=fit[bi],pop[bi].copy()
                l1.mean = 0.7*l1.mean+0.3*pop[bi]
        l1.all_solutions.extend(de_sols)
        return de_sols

    def inject_rk4(self, l1, ns=5):
        self.algo_families.add("rk4_gradient"); self.interventions.append("RK4_inject")
        sols, dim = [], l1.dim
        def gf(t,y):
            eps=1e-5; g=np.zeros(dim); f0=l1.objective(y)
            for i in range(dim): yp=y.copy(); yp[i]+=eps; g[i]=(l1.objective(yp)-f0)/eps
            return -0.1*g
        for _ in range(ns):
            x0=np.clip(l1.mean+RNG.randn(dim)*l1.sigma, l1.bounds[0], l1.bounds[1])
            final=np.clip(rk4_integrate(gf,x0,(0,2),20)[-1], l1.bounds[0], l1.bounds[1])
            sols.append(final); l1.all_solutions.append(final.copy())
            f=l1.objective(final)
            if f<l1.best_ever_fitness: l1.best_ever_fitness,l1.best_ever_x=f,final.copy()
        return sols

    def intervene(self, l1, nov):
        if not self.is_expanding() or self.is_rearranging():
            if len(self.memory_designs)<8: return f"synth_{self.synthesize_new_design().name}"
            if "differential_evolution" not in self.algo_families:
                for s in self.inject_de(l1): nov.register(s)
                return "injected_DE"
            if "rk4_gradient" not in self.algo_families:
                for s in self.inject_rk4(l1): nov.register(s)
                return "injected_RK4"
            self.algo_families.clear()
            for s in self.inject_de(l1,50,30): nov.register(s)
            return "re_injected_DE"
        return "none"

class CMAESMaestroALMA:
    """Three-layer optimizer: CMA-ES + MAESTRO Conductor + ALMA Memory Search."""
    def __init__(self, objective, dim=5, bounds=(-5.12,5.12), max_gen=30):
        self.base_obj, self.dim, self.bounds, self.max_gen = objective, dim, bounds, max_gen
        self.l1 = CMAESLayer(objective, dim, bounds)
        self.cond = MAESTROConductor()
        self.mem = ALMAMemorySearch()
        self.nov = NoveltyDetector()
        self.log = []

    def run(self, verbose=True):
        if verbose:
            print("="*70)
            print("CMAES-MAESTRO-ALMA: Three-Layer Self-Restructuring Optimizer")
            print(f"Objective: {self.base_obj.__name__}, Dim: {self.dim}, Bounds: {self.bounds}")
            print("-"*70)
        for gen in range(self.max_gen):
            ctx = self.cond._get_context(self.l1.history, self.mem.diameter_history,
                len(self.nov.archive), self.l1.get_covariance_condition(), gen, len(self.l1.all_solutions))
            w = self.cond.get_scalarization_weights(ctx)
            pos = self.l1.get_all_positions()
            self.l1.objective = (self.cond.build_composite_objective(self.base_obj, pos, w)
                                 if gen>3 and len(pos)>5 else self.base_obj)
            self.l1.step()
            pos = self.l1.get_all_positions()
            bx, _ = self.l1.get_best()
            tf = self.base_obj(bx) if bx is not None else float("inf")
            for p in pos[-10:]:
                if self.nov.is_genuinely_novel(p): self.nov.register(p)
            self.mem.record_metrics(pos)
            d, v = self.mem.diameter_history[-1], self.mem.volume_history[-1]
            fi = self.l1.history[-2]-self.l1.history[-1] if len(self.l1.history)>=2 else 0
            self.cond.update_conductor(ctx, w, fi+0.01*len(self.nov.archive))
            intv = "none"
            if gen>8:
                a = self.mem.intervene(self.l1, self.nov)
                if a != "none": intv = a
            if gen>0 and gen%8==0: self.l1.objective = self.base_obj
            cc = self.l1.get_covariance_condition()
            self.log.append(dict(gen=gen, fitness=tf, sigma=self.l1.sigma, cond_c=cc,
                diam=d, vol=v, novel=len(self.nov.archive), intv=intv,
                designs=len(self.mem.memory_designs)))
            if verbose and gen%5==0:
                print(f"Gen {gen:3d} | Fit:{tf:10.4f} | Sig:{self.l1.sigma:8.4f} | "
                      f"CC:{cc:8.2f} | D:{d:8.3f} | N:{len(self.nov.archive):4d} | "
                      f"Des:{len(self.mem.memory_designs)} | {intv}")
        bx, _ = self.l1.get_best()
        tb = self.base_obj(bx) if bx is not None else float("inf")
        r = dict(best_x=bx, best_fitness=tb, sigma=self.l1.sigma,
            cond_c=self.l1.get_covariance_condition(),
            diameter=self.mem.diameter_history[-1], volume=self.mem.volume_history[-1],
            novel=len(self.nov.archive), cond_updates=len(self.cond.transformations_applied),
            designs=len(self.mem.memory_designs), interventions=self.mem.interventions,
            families=self.mem.algo_families, log=self.log)
        if verbose:
            print("-"*70)
            print(f"RESULT: fit={tb:.6f} sig={r['sigma']:.4f} CC={r['cond_c']:.2f} "
                  f"diam={r['diameter']:.4f} novel={r['novel']} designs={r['designs']} "
                  f"families={r['families']}")
            print("="*70)
        return r

# ======================== TESTS ========================
def test_rk4_integration():
    err = abs(rk4_integrate(lambda t,y: -y, np.array([1.0]), (0,1), 100)[-1][0] - np.exp(-1))
    assert err < 1e-6; print(f"  test_rk4_integration PASSED (err={err:.2e})")

def test_search_space_diameter():
    d = search_space_diameter([np.array([0,0]), np.array([3,4])])
    assert abs(d-5)<1e-10 and search_space_diameter([np.array([1])])==0
    print(f"  test_search_space_diameter PASSED (d={d})")

def test_novelty_detector():
    nd = NoveltyDetector(3, 0.05); nd.register(np.array([1.,2.,3.]))
    assert not nd.is_genuinely_novel(np.array([1.,2.,3.]))
    assert nd.is_genuinely_novel(np.array([100.,200.,300.]))
    print("  test_novelty_detector PASSED")

def test_cmaes_basic():
    s=RNG.get_state(); RNG.seed(42)
    cma=CMAESLayer(rastrigin,3); [cma.step() for _ in range(15)]
    _,bf=cma.get_best(); assert bf<80
    print(f"  test_cmaes_basic PASSED (fit={bf:.4f})"); RNG.set_state(s)

def test_cmaes_covariance_adapts():
    s=RNG.get_state(); RNG.seed(123)
    cma=CMAESLayer(rosenbrock,3,(-5,10)); C0=cma.C.copy()
    [cma.step() for _ in range(20)]
    diff=np.linalg.norm(cma.C-C0); assert diff>0.01
    print(f"  test_cmaes_covariance_adapts PASSED (diff={diff:.4f})"); RNG.set_state(s)

def test_maestro_conductor():
    c=MAESTROConductor(); ctx=np.array([.1,.5,-.2,1,2,.05])
    w=c.get_scalarization_weights(ctx)
    assert len(w)==4 and abs(sum(w)-1)<0.01 and all(wi>=0 for wi in w)
    c.update_conductor(ctx,w,1.0)
    print(f"  test_maestro_conductor PASSED (w={w.round(3)})")

def test_alma_memory_designs():
    a=ALMAMemorySearch(); assert len(a.memory_designs)==3
    arc=[RNG.randn(3) for _ in range(20)]
    for d in a.memory_designs: assert len(d.retrieve_fn(arc,np.zeros(3),3))<=3
    a.synthesize_new_design(); assert len(a.memory_designs)==4
    print(f"  test_alma_memory_designs PASSED ({len(a.memory_designs)} designs)")

def test_differential_evolution():
    s=RNG.get_state(); RNG.seed(99)
    cma=CMAESLayer(rastrigin,3); cma.step()
    a=ALMAMemorySearch(); de=a.inject_de(cma,10,5)
    assert len(de)>0 and "differential_evolution" in a.algo_families
    print(f"  test_differential_evolution PASSED ({len(de)} sols)"); RNG.set_state(s)

def test_expansion_monitor():
    em=ALMAMemorySearch(stag_win=3)
    for i in range(5): em.record_metrics([np.array([0,0]),np.array([float(i),float(i)])])
    assert em.is_expanding()
    for _ in range(5): em.record_metrics([np.array([0,0]),np.array([5.,5.])])
    assert not em.is_expanding()
    print("  test_expansion_monitor PASSED")

def test_full_system():
    s=RNG.get_state(); RNG.seed(42)
    sys=CMAESMaestroALMA(rastrigin,4,max_gen=15); r=sys.run(False)
    assert r["best_fitness"]<100 and r["novel"]>0 and r["cond_updates"]>0
    print(f"  test_full_system PASSED (fit={r['best_fitness']:.4f} novel={r['novel']} des={r['designs']})")
    RNG.set_state(s)

def run_all_tests():
    print("\nRunning tests..."); print("-"*40)
    test_rk4_integration(); test_search_space_diameter(); test_novelty_detector()
    test_cmaes_basic(); test_cmaes_covariance_adapts(); test_maestro_conductor()
    test_alma_memory_designs(); test_differential_evolution()
    test_expansion_monitor(); test_full_system()
    print("-"*40); print("All tests passed!\n")

def main():
    run_all_tests()
    for name,obj,dim,bounds in [("Rastrigin 5D",rastrigin,5,(-5.12,5.12)),
                                 ("Rosenbrock 4D",rosenbrock,4,(-5,10)),
                                 ("Ackley 6D",ackley,6,(-5,5))]:
        print(f"\n{'='*70}\nDEMO: {name}\n{'='*70}")
        RNG.seed(GLOBAL_SEED)
        CMAESMaestroALMA(obj,dim,bounds,max_gen=30).run(True)

if __name__ == "__main__":
    main()
