"""
DGM-MAP-SA: Darwin-Goedel-Machine + MAP-Elites + Simulated Annealing
=====================================================================
A three-layer self-restructuring optimization system combining ideas from:

Source 1: Darwin Goedel Machine (Zhang et al., arXiv:2505.22954)
  - Open-ended archive of diverse agents, self-improving via variation
  Source 2: MAP-Elites / Quality-Diversity (Mouret & Clune, 2015)
    - Behavior-space archive maintaining diverse high-quality solutions
    Source 3: Simulated Annealing (Kirkpatrick et al., 1983) + Memetic Algorithms
      - Cross-domain metaheuristic transfer from Wikipedia's metaheuristic families

      Layer 1: Solves optimization via MAP-Elites style quality-diversity search
      Layer 2: Restructures the optimization (loss functions, topology, state vars)
      Layer 3: Monitors whether Layer 2 genuinely expands reachable space;
               if stuck, pulls in structurally different components (SA, PSO)

               Includes:
               - Search-space-diameter metric
               - Novelty detector for genuinely new strategies
               - Cross-domain transfer between algorithm families
               - Real numerical integration (RK4)
               - Fixed random seeds
               - Toy demo in main()
               - Test functions
               """

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import entropy as kl_divergence_proxy
from dataclasses import dataclass, field
from typing import List, Callable, Dict, Tuple, Optional
import copy
import hashlib
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Fixed random seeds for reproducibility
# ---------------------------------------------------------------------------
GLOBAL_SEED = 42
RNG = np.random.RandomState(GLOBAL_SEED)


# ---------------------------------------------------------------------------
# Utility: Runge-Kutta 4th order integrator
# ---------------------------------------------------------------------------
def rk4_step(f: Callable, y: np.ndarray, t: float, dt: float) -> np.ndarray:
      """Single step of RK4 integration for dy/dt = f(t, y)."""
      k1 = dt * f(t, y)
      k2 = dt * f(t + dt / 2.0, y + k1 / 2.0)
      k3 = dt * f(t + dt / 2.0, y + k2 / 2.0)
      k4 = dt * f(t + dt, y + k3)
      return y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def rk4_integrate(f: Callable, y0: np.ndarray, t_span: Tuple[float, float],
                                    n_steps: int = 100) -> np.ndarray:
                                          """Integrate dy/dt = f(t, y) from t_span[0] to t_span[1] using RK4."""
                                          t0, tf = t_span
                                          dt = (tf - t0) / n_steps
                                          y = y0.copy()
                                          t = t0
                                          trajectory = [y.copy()]
                                          for _ in range(n_steps):
                                                    y = rk4_step(f, y, t, dt)
                                                    t += dt
                                                    trajectory.append(y.copy())
                                                return np.array(trajectory)


# ---------------------------------------------------------------------------
# Search-space-diameter metric
# ---------------------------------------------------------------------------
def search_space_diameter(archive: List[np.ndarray]) -> float:
      """Compute the diameter of the solution set (max pairwise distance)."""
    if len(archive) < 2:
              return 0.0
    pts = np.array(archive)
    dists = cdist(pts, pts, metric='euclidean')
    return float(np.max(dists))


def search_space_volume_proxy(archive: List[np.ndarray]) -> float:
      """Estimate volume via convex hull proxy (std dev product)."""
    if len(archive) < 2:
              return 0.0
    pts = np.array(archive)
    stds = np.std(pts, axis=0)
    return float(np.prod(stds + 1e-12))


# ---------------------------------------------------------------------------
# Novelty detector
# ---------------------------------------------------------------------------
class NoveltyDetector:
      """Detects genuinely new strategies vs recombinations of known ones."""

    def __init__(self, k_nearest: int = 5, novelty_threshold: float = 0.1):
              self.k_nearest = k_nearest
        self.novelty_threshold = novelty_threshold
        self.seen_hashes: set = set()
        self.archive: List[np.ndarray] = []

    def _structural_hash(self, solution: np.ndarray) -> str:
              quantized = np.round(solution, decimals=2)
        return hashlib.md5(quantized.tobytes()).hexdigest()

    def novelty_score(self, candidate: np.ndarray) -> float:
              if len(self.archive) < self.k_nearest:
                            return float('inf')
        pts = np.array(self.archive)
        dists = np.linalg.norm(pts - candidate, axis=1)
        k = min(self.k_nearest, len(dists))
        nearest_dists = np.sort(dists)[:k]
        return float(np.mean(nearest_dists))

    def is_genuinely_novel(self, candidate: np.ndarray) -> bool:
              h = self._structural_hash(candidate)
        if h in self.seen_hashes:
                      return False
        score = self.novelty_score(candidate)
        return score > self.novelty_threshold

    def register(self, candidate: np.ndarray):
              h = self._structural_hash(candidate)
        self.seen_hashes.add(h)
        self.archive.append(candidate.copy())


# ---------------------------------------------------------------------------
# Objective functions (toy problems for demonstration)
# ---------------------------------------------------------------------------
def rastrigin(x: np.ndarray) -> float:
      A = 10
    return float(A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x)))


def rosenbrock(x: np.ndarray) -> float:
      return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))


def ackley(x: np.ndarray) -> float:
      n = len(x)
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(2 * np.pi * x))
    return float(-20 * np.exp(-0.2 * np.sqrt(s1 / n))
                                  - np.exp(s2 / n) + 20 + np.e)


# ---------------------------------------------------------------------------
# Layer 1: MAP-Elites style Quality-Diversity optimizer
# (Inspired by MAP-Elites from Quality-Diversity literature)
# ---------------------------------------------------------------------------
@dataclass
class Solution:
      x: np.ndarray
    fitness: float
    behavior: np.ndarray  # behavior descriptor for MAP-Elites archive


class MapElitesLayer:
      """Layer 1: Quality-diversity optimization via MAP-Elites."""

    def __init__(self, objective: Callable, dim: int, n_bins: int = 10,
                                  bounds: Tuple[float, float] = (-5.12, 5.12),
                                  mutation_sigma: float = 0.5):
                                            self.objective = objective
                                            self.dim = dim
                                            self.n_bins = n_bins
                                            self.bounds = bounds
                                            self.mutation_sigma = mutation_sigma
                                            self.archive: Dict[tuple, Solution] = {}
                                            self.history: List[float] = []
                                            self.all_solutions: List[np.ndarray] = []

    def _behavior_descriptor(self, x: np.ndarray) -> np.ndarray:
              clipped = np.clip(x[:2], self.bounds[0], self.bounds[1])
        normalized = (clipped - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
        return normalized

    def _discretize(self, bd: np.ndarray) -> tuple:
              bins = np.clip((bd * self.n_bins).astype(int), 0, self.n_bins - 1)
        return tuple(bins)

    def _random_solution(self) -> np.ndarray:
              return RNG.uniform(self.bounds[0], self.bounds[1], self.dim)

    def _mutate(self, x: np.ndarray) -> np.ndarray:
              noise = RNG.randn(self.dim) * self.mutation_sigma
        child = x + noise
        return np.clip(child, self.bounds[0], self.bounds[1])

    def step(self, n_evals: int = 50) -> float:
              best_fitness = float('inf')
        for _ in range(n_evals):
                      if len(self.archive) < 10 or RNG.rand() < 0.2:
                                        x = self._random_solution()
else:
                      parent = list(self.archive.values())[
                    RNG.randint(len(self.archive))]
                x = self._mutate(parent.x)

            fitness = self.objective(x)
            bd = self._behavior_descriptor(x)
            cell = self._discretize(bd)

            if cell not in self.archive or fitness < self.archive[cell].fitness:
                              self.archive[cell] = Solution(x=x.copy(), fitness=fitness,
                                                                                                          behavior=bd)

            self.all_solutions.append(x.copy())
            best_fitness = min(best_fitness, fitness)

        self.history.append(best_fitness)
        return best_fitness

    def get_best(self) -> Optional[Solution]:
              if not self.archive:
                            return None
        return min(self.archive.values(), key=lambda s: s.fitness)

    def get_all_positions(self) -> List[np.ndarray]:
              return [s.x for s in self.archive.values()]


# ---------------------------------------------------------------------------
# Layer 2: Meta-optimizer that restructures the search
# (Inspired by Darwin Goedel Machine's self-modifying approach)
# ---------------------------------------------------------------------------
class MetaRestructurer:
      """Layer 2: Watches Layer 1 and restructures the optimization itself.

          Modifies loss functions, state variables, topology -- not just
              hyperparameters. Inspired by the DGM's approach of iteratively
                  modifying its own code and empirically validating changes.
                      """

    def __init__(self):
              self.transformations_applied: List[str] = []
        self.performance_deltas: List[float] = []
        self.current_loss_transform: Optional[Callable] = None
        self.dim_expansion_count: int = 0
        self.topology_changes: int = 0

    def analyze_stagnation(self, history: List[float],
                                                      window: int = 5) -> bool:
                                                                if len(history) < window:
                                                                              return False
                                                                          recent = history[-window:]
        improvement = abs(recent[0] - recent[-1])
        return improvement < 1e-4

    def restructure_loss(self, base_objective: Callable,
                                                  archive_positions: List[np.ndarray]) -> Callable:
                                                            """Add a novelty-seeking term to the loss function."""
        def augmented_loss(x: np.ndarray) -> float:
                      base = base_objective(x)
            if len(archive_positions) > 1:
                              pts = np.array(archive_positions)
                dists = np.linalg.norm(pts - x, axis=1)
                novelty_bonus = -0.1 * np.mean(np.sort(dists)[:3])
else:
                novelty_bonus = 0.0
            return base + novelty_bonus

        self.transformations_applied.append("loss_augmentation_novelty")
        return augmented_loss

    def expand_state_space(self, layer1: MapElitesLayer,
                                                      extra_dims: int = 2) -> MapElitesLayer:
                                                                """Expand the dimensionality of the search space."""
        old_dim = layer1.dim
        new_dim = old_dim + extra_dims
        layer1.dim = new_dim

        new_archive = {}
        for cell, sol in layer1.archive.items():
                      new_x = np.zeros(new_dim)
            new_x[:old_dim] = sol.x
            new_x[old_dim:] = RNG.randn(extra_dims) * 0.1
            new_archive[cell] = Solution(
                              x=new_x,
                              fitness=sol.fitness,
                              behavior=sol.behavior
            )
        layer1.archive = new_archive
        self.dim_expansion_count += 1
        self.transformations_applied.append(
                      f"dim_expansion_{old_dim}->{new_dim}")
        return layer1

    def change_topology(self, layer1: MapElitesLayer) -> MapElitesLayer:
              """Change the behavior-space binning (topology of the archive)."""
        old_bins = layer1.n_bins
        layer1.n_bins = max(3, old_bins + RNG.choice([-2, 2]))
        layer1.archive.clear()
        self.topology_changes += 1
        self.transformations_applied.append(
                      f"topology_rebinning_{old_bins}->{layer1.n_bins}")
        return layer1

    def apply_rk4_dynamics_loss(self, base_objective: Callable) -> Callable:
              """Reformulate the loss using RK4-integrated gradient flow dynamics.

                      Instead of evaluating f(x) directly, simulate a gradient-like
                     dynamical system and evaluate the endpoint.
                             """
        def dynamics_loss(x: np.ndarray) -> float:
                      dim = len(x)

            def gradient_flow(t, y):
                              eps = 1e-5
                grad = np.zeros(dim)
                f0 = base_objective(y)
                for i in range(dim):
                                      y_plus = y.copy()
                    y_plus[i] += eps
                    grad[i] = (base_objective(y_plus) - f0) / eps
                return -0.1 * grad  # gradient descent dynamics

            trajectory = rk4_integrate(gradient_flow, x, (0, 1.0),
                                                                              n_steps=10)
            return float(base_objective(trajectory[-1]))

        self.transformations_applied.append("rk4_dynamics_loss")
        return dynamics_loss


# ---------------------------------------------------------------------------
# Layer 3: Expansion monitor + cross-domain algorithm transfer
# (Monitors whether Layer 2 is genuinely expanding or just rearranging)
# ---------------------------------------------------------------------------
class ExpansionMonitor:
      """Layer 3: Monitors whether the reachable space is genuinely growing.

          If stuck, pulls in structurally different components from another
              algorithm family (e.g., Simulated Annealing from the metaheuristic
                  family, or PSO from swarm intelligence).
                      """

    def __init__(self, stagnation_window: int = 5,
                                  expansion_threshold: float = 0.01):
                                            self.diameter_history: List[float] = []
        self.volume_history: List[float] = []
        self.stagnation_window = stagnation_window
        self.expansion_threshold = expansion_threshold
        self.interventions: List[str] = []
        self.algorithm_families_used: set = set()

    def record_metrics(self, archive_positions: List[np.ndarray]):
              d = search_space_diameter(archive_positions)
        v = search_space_volume_proxy(archive_positions)
        self.diameter_history.append(d)
        self.volume_history.append(v)

    def is_genuinely_expanding(self) -> bool:
              if len(self.diameter_history) < self.stagnation_window:
                            return True
        recent = self.diameter_history[-self.stagnation_window:]
        growth = (recent[-1] - recent[0]) / (abs(recent[0]) + 1e-12)
        return growth > self.expansion_threshold

    def is_just_rearranging(self) -> bool:
              if len(self.volume_history) < self.stagnation_window:
                            return False
        recent_vol = self.volume_history[-self.stagnation_window:]
        recent_dia = self.diameter_history[-self.stagnation_window:]
        vol_change = abs(recent_vol[-1] - recent_vol[0]) / (
                      abs(recent_vol[0]) + 1e-12)
        dia_change = abs(recent_dia[-1] - recent_dia[0]) / (
                      abs(recent_dia[0]) + 1e-12)
        return vol_change < self.expansion_threshold and \
            dia_change < self.expansion_threshold

    # --- Cross-domain algorithm transfers ---

    def inject_simulated_annealing(self, layer1: MapElitesLayer,
                                                                      n_steps: int = 100,
                                                                      t_init: float = 10.0,
                                                                      t_min: float = 0.01) -> List[np.ndarray]:
                                                                                """Pull in Simulated Annealing from the metaheuristic family."""
                                                                                self.algorithm_families_used.add("simulated_annealing")
                                                                                self.interventions.append("cross_domain_SA_injection")

        best = layer1.get_best()
        x = best.x.copy() if best else RNG.uniform(
                      layer1.bounds[0], layer1.bounds[1], layer1.dim)
        current_e = layer1.objective(x)
        sa_solutions = [x.copy()]
        temp = t_init

        for i in range(n_steps):
                      neighbor = x + RNG.randn(layer1.dim) * 0.5
            neighbor = np.clip(neighbor, layer1.bounds[0], layer1.bounds[1])
            neighbor_e = layer1.objective(neighbor)
            delta = neighbor_e - current_e

            if delta < 0 or RNG.rand() < np.exp(-delta / (temp + 1e-12)):
                              x = neighbor
                current_e = neighbor_e
                sa_solutions.append(x.copy())

                bd = layer1._behavior_descriptor(x)
                cell = layer1._discretize(bd)
                if (cell not in layer1.archive or
                                            current_e < layer1.archive[cell].fitness):
                                                                  layer1.archive[cell] = Solution(
                                                                                            x=x.copy(), fitness=current_e, behavior=bd)

            temp *= 0.95  # geometric cooling

        return sa_solutions

    def inject_pso(self, layer1: MapElitesLayer,
                                      n_particles: int = 20,
                                      n_steps: int = 50) -> List[np.ndarray]:
                                                """Pull in Particle Swarm Optimization from swarm intelligence."""
        self.algorithm_families_used.add("particle_swarm_optimization")
        self.interventions.append("cross_domain_PSO_injection")

        dim = layer1.dim
        lo, hi = layer1.bounds

        positions = RNG.uniform(lo, hi, (n_particles, dim))
        velocities = RNG.randn(n_particles, dim) * 0.1
        p_best = positions.copy()
        p_best_scores = np.array([layer1.objective(p) for p in positions])
        g_best_idx = np.argmin(p_best_scores)
        g_best = p_best[g_best_idx].copy()
        all_positions = [g_best.copy()]

        w, c1, c2 = 0.7, 1.5, 1.5
        for _ in range(n_steps):
                      r1 = RNG.rand(n_particles, dim)
            r2 = RNG.rand(n_particles, dim)
            velocities = (w * velocities
                                                    + c1 * r1 * (p_best - positions)
                                                    + c2 * r2 * (g_best - positions))
            positions = np.clip(positions + velocities, lo, hi)

            for i in range(n_particles):
                              score = layer1.objective(positions[i])
                if score < p_best_scores[i]:
                                      p_best[i] = positions[i].copy()
                    p_best_scores[i] = score

                bd = layer1._behavior_descriptor(positions[i])
                cell = layer1._discretize(bd)
                if (cell not in layer1.archive or
                                            score < layer1.archive[cell].fitness):
                                                                  layer1.archive[cell] = Solution(
                                                                                            x=positions[i].copy(), fitness=score, behavior=bd)

            g_best_idx = np.argmin(p_best_scores)
            g_best = p_best[g_best_idx].copy()
            all_positions.append(g_best.copy())

        return all_positions

    def intervene(self, layer1: MapElitesLayer,
                                    novelty_detector: NoveltyDetector) -> str:
                                              """Decide which cross-domain transfer to apply."""
        if not self.is_genuinely_expanding() or self.is_just_rearranging():
                      if "simulated_annealing" not in self.algorithm_families_used:
                                        sa_sols = self.inject_simulated_annealing(layer1)
                for s in sa_sols:
                                      novelty_detector.register(s)
                return "injected_simulated_annealing"
elif ("particle_swarm_optimization" not in
                        self.algorithm_families_used):
                pso_sols = self.inject_pso(layer1)
                for s in pso_sols:
                                      novelty_detector.register(s)
                return "injected_pso"
else:
                # Reset and re-inject with different parameters
                  self.algorithm_families_used.clear()
                sa_sols = self.inject_simulated_annealing(
                                      layer1, n_steps=200, t_init=50.0)
                for s in sa_sols:
                                      novelty_detector.register(s)
                return "re_injected_SA_with_higher_temp"
        return "no_intervention_needed"


# ---------------------------------------------------------------------------
# Main orchestrator: DGM-MAP-SA system
# ---------------------------------------------------------------------------
class DGMMapSA:
      """Three-layer self-restructuring optimizer.

          Named after its sources:
              DGM  = Darwin Goedel Machine (open-ended self-improvement)
                  MAP  = MAP-Elites (quality-diversity)
                      SA   = Simulated Annealing + PSO (cross-domain metaheuristic transfer)
                          """

    def __init__(self, objective: Callable, dim: int = 5,
                                  bounds: Tuple[float, float] = (-5.12, 5.12),
                                  max_generations: int = 30):
                                            self.base_objective = objective
        self.dim = dim
        self.bounds = bounds
        self.max_generations = max_generations

        # Layer 1
        self.layer1 = MapElitesLayer(
                      objective=objective, dim=dim, bounds=bounds)

        # Layer 2
        self.meta = MetaRestructurer()

        # Layer 3
        self.monitor = ExpansionMonitor()

        # Novelty detection
        self.novelty = NoveltyDetector()

        # Tracking
        self.generation_log: List[dict] = []

    def run(self, verbose: bool = True) -> dict:
              if verbose:
                            print("=" * 65)
            print("DGM-MAP-SA: Three-Layer Self-Restructuring Optimizer")
            print("=" * 65)
            print(f"Objective: {self.base_objective.__name__}")
            print(f"Dimensions: {self.dim}, Bounds: {self.bounds}")
            print(f"Max generations: {self.max_generations}")
            print("-" * 65)

        for gen in range(self.max_generations):
                      # --- Layer 1: Run MAP-Elites ---
                      best_fitness = self.layer1.step(n_evals=50)
            positions = self.layer1.get_all_positions()

            # Register with novelty detector
            for p in positions[-10:]:
                              if self.novelty.is_genuinely_novel(p):
                                                    self.novelty.register(p)

            # --- Layer 3: Monitor expansion ---
            self.monitor.record_metrics(positions)
            diameter = self.monitor.diameter_history[-1]
            volume = self.monitor.volume_history[-1]

            # --- Layer 2: Check for stagnation, restructure if needed ---
            intervention = "none"
            if gen > 5 and self.meta.analyze_stagnation(
                                  self.layer1.history):
                                                    # First try restructuring the loss
                                                    if gen % 3 == 0:
                                                                          new_obj = self.meta.restructure_loss(
                                                                                                    self.base_objective, positions)
                                                                          self.layer1.objective = new_obj
                                                                          intervention = "loss_restructured"
elif gen % 3 == 1:
                    new_obj = self.meta.apply_rk4_dynamics_loss(
                                              self.base_objective)
                    self.layer1.objective = new_obj
                    intervention = "rk4_dynamics_loss"
else:
                    self.layer1 = self.meta.change_topology(self.layer1)
                    intervention = "topology_changed"

            # --- Layer 3: Check if Layer 2 is genuinely expanding ---
            if gen > 8:
                              l3_action = self.monitor.intervene(
                                                    self.layer1, self.novelty)
                if l3_action != "no_intervention_needed":
                                      intervention = l3_action

            # Periodically reset objective to base (avoid drift)
            if gen > 0 and gen % 10 == 0:
                              self.layer1.objective = self.base_objective

            best = self.layer1.get_best()
            gen_info = {
                              "generation": gen,
                              "best_fitness": best.fitness if best else float('inf'),
                              "archive_size": len(self.layer1.archive),
                              "diameter": diameter,
                              "volume": volume,
                              "novel_count": len(self.novelty.archive),
                              "intervention": intervention,
                              "dim": self.layer1.dim,
            }
            self.generation_log.append(gen_info)

            if verbose and gen % 5 == 0:
                              print(
                                                    f"Gen {gen:3d} | "
                                                    f"Best: {gen_info['best_fitness']:10.4f} | "
                                                    f"Archive: {gen_info['archive_size']:4d} | "
                                                    f"Diameter: {gen_info['diameter']:8.3f} | "
                                                    f"Novel: {gen_info['novel_count']:4d} | "
                                                    f"Dim: {gen_info['dim']} | "
                                                    f"Action: {intervention}"
                              )

        best = self.layer1.get_best()
        result = {
                      "best_solution": best.x if best else None,
                      "best_fitness": best.fitness if best else float('inf'),
                      "final_archive_size": len(self.layer1.archive),
                      "final_diameter": self.monitor.diameter_history[-1],
                      "final_volume": self.monitor.volume_history[-1],
                      "total_novel_strategies": len(self.novelty.archive),
                      "transformations": self.meta.transformations_applied,
                      "interventions": self.monitor.interventions,
                      "algorithm_families_used": self.monitor.algorithm_families_used,
                      "generation_log": self.generation_log,
        }

        if verbose:
                      print("-" * 65)
            print("RESULTS:")
            print(f"  Best fitness:        {result['best_fitness']:.6f}")
            print(f"  Final archive size:  {result['final_archive_size']}")
            print(f"  Search diameter:     {result['final_diameter']:.4f}")
            print(f"  Novel strategies:    {result['total_novel_strategies']}")
            print(f"  Transformations:     {result['transformations']}")
            print(f"  L3 interventions:    {result['interventions']}")
            print(f"  Algorithm families:  {result['algorithm_families_used']}")
            print("=" * 65)

        return result


# ===========================================================================
# Test functions
# ===========================================================================

def test_rk4_integration():
      """Test RK4 integrator on a known ODE: dy/dt = -y, y(0) = 1."""
    def decay(t, y):
        return -y
    y0 = np.array([1.0])
    traj = rk4_integrate(decay, y0, (0, 1.0), n_steps=100)
    final = traj[-1][0]
    expected = np.exp(-1.0)
    error = abs(final - expected)
    assert error < 1e-6, f"RK4 error too large: {error}"
    print(f"  test_rk4_integration PASSED (error={error:.2e})")


def test_search_space_diameter():
      """Test diameter metric on known point sets."""
    pts = [np.array([0, 0]), np.array([3, 4])]
    d = search_space_diameter(pts)
    assert abs(d - 5.0) < 1e-10, f"Expected 5.0, got {d}"
    assert search_space_diameter([np.array([1])]) == 0.0
    print(f"  test_search_space_diameter PASSED (d={d})")


def test_novelty_detector():
      """Test that identical solutions are not considered novel."""
    nd = NoveltyDetector(k_nearest=3, novelty_threshold=0.05)
    x = np.array([1.0, 2.0, 3.0])
    nd.register(x)
    assert not nd.is_genuinely_novel(x), "Identical should not be novel"
    far = np.array([100.0, 200.0, 300.0])
    # With only 1 point in archive and k=3, novelty_score returns inf
    assert nd.is_genuinely_novel(far), "Far point should be novel"
    print("  test_novelty_detector PASSED")


def test_map_elites_basic():
      """Test that MAP-Elites finds improving solutions."""
    rng_state = RNG.get_state()
    RNG.seed(42)
    me = MapElitesLayer(objective=rastrigin, dim=3, bounds=(-5.12, 5.12))
    for _ in range(10):
              me.step(n_evals=20)
    best = me.get_best()
    assert best is not None, "Should have at least one solution"
    assert best.fitness < 100, f"Fitness too high: {best.fitness}"
    print(f"  test_map_elites_basic PASSED (fitness={best.fitness:.4f})")
    RNG.set_state(rng_state)


def test_meta_restructurer():
      """Test that meta-restructurer detects stagnation."""
    mr = MetaRestructurer()
    flat_history = [10.0, 10.0, 10.0, 10.0, 10.0]
    assert mr.analyze_stagnation(flat_history), "Should detect stagnation"
    improving = [10.0, 8.0, 5.0, 2.0, 0.5]
    assert not mr.analyze_stagnation(improving), "Should not flag improvement"
    print("  test_meta_restructurer PASSED")


def test_expansion_monitor():
      """Test expansion monitor detects when space stops growing."""
    em = ExpansionMonitor(stagnation_window=3, expansion_threshold=0.01)
    for i in range(5):
              pts = [np.array([0, 0]), np.array([float(i), float(i)])]
        em.record_metrics(pts)
    assert em.is_genuinely_expanding(), "Should be expanding"

    for _ in range(5):
              pts = [np.array([0, 0]), np.array([5.0, 5.0])]
        em.record_metrics(pts)
    assert not em.is_genuinely_expanding(), "Should detect stagnation"
    print("  test_expansion_monitor PASSED")


def test_cross_domain_transfer():
      """Test that SA and PSO injection work and add solutions."""
    rng_state = RNG.get_state()
    RNG.seed(99)
    me = MapElitesLayer(objective=rastrigin, dim=3, bounds=(-5.12, 5.12))
    me.step(n_evals=20)
    em = ExpansionMonitor()

    sa_sols = em.inject_simulated_annealing(me, n_steps=20)
    assert len(sa_sols) > 0, "SA should produce solutions"
    assert "simulated_annealing" in em.algorithm_families_used

    pso_sols = em.inject_pso(me, n_particles=5, n_steps=10)
    assert len(pso_sols) > 0, "PSO should produce solutions"
    assert "particle_swarm_optimization" in em.algorithm_families_used
    print(f"  test_cross_domain_transfer PASSED "
                    f"(SA: {len(sa_sols)} sols, PSO: {len(pso_sols)} sols)")
    RNG.set_state(rng_state)


def test_full_system():
      """Integration test: run full DGM-MAP-SA on Rastrigin."""
    rng_state = RNG.get_state()
    RNG.seed(42)
    system = DGMMapSA(
              objective=rastrigin, dim=4, max_generations=15)
    result = system.run(verbose=False)
    assert result['best_fitness'] < 50, \
        f"System should find reasonable solution, got {result['best_fitness']}"
    assert result['final_archive_size'] > 0
    assert result['total_novel_strategies'] > 0
    print(f"  test_full_system PASSED "
                    f"(fitness={result['best_fitness']:.4f}, "
                    f"archive={result['final_archive_size']}, "
                    f"novel={result['total_novel_strategies']})")
    RNG.set_state(rng_state)


def run_all_tests():
      print("\nRunning tests...")
    print("-" * 40)
    test_rk4_integration()
    test_search_space_diameter()
    test_novelty_detector()
    test_map_elites_basic()
    test_meta_restructurer()
    test_expansion_monitor()
    test_cross_domain_transfer()
    test_full_system()
    print("-" * 40)
    print("All tests passed!\n")


# ===========================================================================
# Main demo
# ===========================================================================

def main():
      """Toy demo of the DGM-MAP-SA three-layer system."""
    run_all_tests()

    print("\n" + "=" * 65)
    print("DEMO 1: Rastrigin function (5D)")
    print("=" * 65)
    RNG.seed(GLOBAL_SEED)
    system1 = DGMMapSA(
              objective=rastrigin, dim=5, max_generations=30)
    result1 = system1.run(verbose=True)

    print("\n" + "=" * 65)
    print("DEMO 2: Rosenbrock function (4D)")
    print("=" * 65)
    RNG.seed(GLOBAL_SEED + 1)
    system2 = DGMMapSA(
              objective=rosenbrock, dim=4, max_generations=30,
              bounds=(-5.0, 10.0))
    result2 = system2.run(verbose=True)

    print("\n" + "=" * 65)
    print("DEMO 3: Ackley function (6D)")
    print("=" * 65)
    RNG.seed(GLOBAL_SEED + 2)
    system3 = DGMMapSA(
              objective=ackley, dim=6, max_generations=30,
              bounds=(-5.0, 5.0))
    result3 = system3.run(verbose=True)

    # Summary of diameter tracking across demos
    print("\n" + "=" * 65)
    print("SEARCH SPACE DIAMETER TRACKING SUMMARY")
    print("=" * 65)
    for name, result in [("Rastrigin", result1),
                                                  ("Rosenbrock", result2),
                                                  ("Ackley", result3)]:
                                                            log = result['generation_log']
        diameters = [g['diameter'] for g in log]
        print(f"  {name:12s}: "
                            f"initial_d={diameters[0]:.3f}, "
                            f"final_d={diameters[-1]:.3f}, "
                            f"max_d={max(diameters):.3f}, "
                            f"families={result['algorithm_families_used']}")


if __name__ == "__main__":
      main()
