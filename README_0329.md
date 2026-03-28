# CMAES-MAESTRO-ALMA: Three-Layer Self-Restructuring Optimizer

## What it does

CMAES-MAESTRO-ALMA is a three-layer optimization system that goes beyond the previous DGM-MAP-SA (0327) by introducing three entirely new algorithmic components. Layer 1 uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy) instead of MAP-Elites, learning a full second-order model of the objective landscape by adapting its covariance matrix — this allows the optimizer to discover and exploit variable correlations rather than relying on grid-based behavior descriptors. Layer 2 replaces the hard-coded heuristic restructurer with a MAESTRO-inspired Conductor that dynamically re-weights multiple objectives (fitness, novelty, diversity, RK4-smoothness) using a contextual bandit with group-relative advantages, effectively learning *which* objective restructuring works best in each optimization context. Layer 3 replaces the fixed SA/PSO injection with an ALMA-inspired open-ended memory design search that maintains an evolving library of retrieval/update strategies and can synthesize new hybrid designs; when expansion stalls, it pulls in Differential Evolution as a structurally different cross-domain algorithm family (instead of the SA and PSO used in 0327).

## Sources inspiring each layer

### Layer 1: CMA-ES (Covariance Matrix Adaptation)
- **CMA-ES** (Hansen & Ostermeier, 2001): Adapts the covariance matrix of a multivariate normal search distribution to learn landscape shape. From the [Wikipedia metaheuristic family](https://en.wikipedia.org/wiki/CMA-ES).

### Layer 2: MAESTRO Dynamic Scalarization Conductor
- **MAESTRO** (Zhao et al., Jan 2026, [arXiv:2601.07208](https://arxiv.org/abs/2601.07208)): Meta-learning Adaptive Estimation of Scalarization Trade-offs for Reward Optimization. A Conductor network co-evolves with the policy via contextual bandit bi-level optimization.

### Layer 3: ALMA Memory Design Search + Differential Evolution
- **ALMA** (Xiong, Hu & Clune, Feb 2026, [arXiv:2602.07755](https://arxiv.org/abs/2602.07755)): Automated meta-Learning of Memory designs for Agentic systems. Searches over memory designs expressed as executable code in an open-ended manner.
- **Differential Evolution** (Storn & Price, 1997): Cross-domain transfer from the evolutionary computation family, structurally different from CMA-ES. From [Wikipedia](https://en.wikipedia.org/wiki/Differential_evolution).

## How to run

\`\`\`bash
pip install numpy scipy
python meta_transfer_0329.py
\`\`\`

Runs all tests first, then three optimization demos (Rastrigin 5D, Rosenbrock 4D, Ackley 6D).

## What's new vs previous files (meta_transfer_0327.py)

| Component | 0327 (DGM-MAP-SA) | 0329 (CMAES-MAESTRO-ALMA) |
|---|---|---|
| Layer 1 optimizer | MAP-Elites (grid archive) | CMA-ES (learned covariance matrix) |
| Layer 2 restructuring | Hard-coded heuristic rules | MAESTRO contextual bandit Conductor |
| Layer 3 cross-domain | SA + PSO injection | Differential Evolution + ALMA memory synthesis |
| Objective weighting | Fixed novelty bonus | Dynamic 4-arm scalarization with learned weights |
| Memory/archive strategy | Fixed behavior descriptors | Evolving library of retrieval/update designs |
| Self-adaptation | Topology rebinning, dim expansion | Covariance matrix adaptation + conductor learning |

## Honest limitations

- **Conductor is shallow**: The MAESTRO-inspired Conductor uses a linear contextual bandit rather than the neural network in the original paper. Real MAESTRO uses terminal hidden states as a semantic bottleneck.
- **Memory designs are hand-seeded**: While the system can synthesize hybrid designs, the initial design library (nearest-neighbor, diversity-preserving, recency-weighted) is hand-crafted. True ALMA would discover designs from scratch.
- **CMA-ES at toy scale**: The demos use 4-6 dimensional problems. CMA-ES scales to hundreds of dimensions but needs more evaluations.
- **No theoretical guarantees**: Neither CMA-ES convergence theory nor bandit regret bounds apply to this composite system.
- **Fixed random seeds**: Results are reproducible but may not represent typical behavior.
- **Composite objective is expensive**: The RK4-smoothness arm in the Conductor evaluates a gradient flow integration at every function call, which is costly.
- **Cross-domain transfer is one-directional**: DE solutions feed into CMA-ES but the covariance information doesn't flow back to DE.
