# DGM-MAP-SA: Three-Layer Self-Restructuring Optimizer

## What it does

DGM-MAP-SA is a three-layer optimization system that goes beyond fixed-architecture search. Layer 1 solves an optimization problem using MAP-Elites quality-diversity search, maintaining an archive of diverse high-performing solutions across a behavior space. Layer 2 watches Layer 1 for stagnation and restructures the optimization itself — modifying loss functions (adding novelty-seeking terms), changing the behavior-space topology, and reformulating evaluations through RK4-integrated gradient flow dynamics. Layer 3 monitors whether Layer 2's changes are genuinely expanding the reachable solution space (via diameter and volume metrics) or merely rearranging existing solutions; when expansion stalls, it injects structurally different algorithm components from other metaheuristic families (Simulated Annealing, Particle Swarm Optimization) to break out of local attractors.

## Sources and inspiration for each layer

### Layer 1: MAP-Elites / Quality-Diversity Search
- **MAP-Elites** (Mouret & Clune, 2015): Maintains a grid-structured archive where each cell holds the best solution for a particular behavior descriptor, promoting diverse high-quality solutions.
- **Open-endedness project** ([Pabloo22/open-endedness-project](https://github.com/Pabloo22/open-endedness-project)): Intrinsic + extrinsic reward merging in RL, using Transformer-XL policies in open-ended environments.

### Layer 2: Self-Modifying Meta-Optimization
- **Darwin Goedel Machine** (Zhang et al., 2025, [arXiv:2505.22954](https://arxiv.org/abs/2505.22954)): A self-improving system that iteratively modifies its own code and validates changes empirically, maintaining an open-ended archive of diverse agents.
- **Automated Continual Learning** (Irie et al., 2023, [arXiv:2312.00276](https://arxiv.org/abs/2312.00276)): Self-referential neural networks that meta-learn their own continual learning algorithms by encoding desiderata into meta-objectives.
- **Surprisal MCTS** ([jbarnes850/surprisal](https://github.com/jbarnes850/surprisal)): Bayesian surprise-guided Monte Carlo tree search for open-ended discovery.

### Layer 3: Cross-Domain Metaheuristic Transfer
- **Simulated Annealing** (Kirkpatrick et al., 1983): Classic stochastic optimization with temperature-controlled acceptance of worse solutions, from the [Wikipedia metaheuristic family overview](https://en.wikipedia.org/wiki/Metaheuristic).
- **Particle Swarm Optimization** (Kennedy & Eberhart, 1995): Swarm intelligence approach where particles share information about good regions.
- **Memetic Algorithms** (Moscato, 1989): The concept of hybridizing population-based search with local improvement from structurally different algorithm families.

## How to run

```bash
pip install numpy scipy
python meta_transfer_0327.py
```

This runs all tests first, then three optimization demos (Rastrigin 5D, Rosenbrock 4D, Ackley 6D), printing progress and a diameter-tracking summary.

## Honest limitations

- **No true self-modification**: Unlike the Darwin Goedel Machine which actually rewrites its own code, this system picks from a pre-defined menu of restructuring operations. The set of possible interventions is hard-coded, not discovered.
- **Toy scale only**: The demos use 4-6 dimensional problems with small evaluation budgets. Real-world problems would need orders of magnitude more computation.
- **Behavior descriptors are naive**: Using the first two coordinates as behavior descriptors is a placeholder. Real quality-diversity needs domain-appropriate descriptors.
- **No learned meta-policy**: Layer 2's decisions (when to restructure, which transformation to apply) follow simple heuristic rules, not a learned meta-policy as in true meta-learning.
- **Cross-domain transfer is shallow**: Injecting SA or PSO solutions into the MAP-Elites archive is helpful but doesn't capture the deep algorithmic synergies that memetic algorithms aim for.
- **No convergence guarantees**: The system has no theoretical guarantees about finding global optima or even about improvement over time.
- **Fixed random seeds affect generalizability**: Results are reproducible but may not represent typical behavior across different random seeds.
