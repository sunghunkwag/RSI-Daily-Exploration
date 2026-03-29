# RSI-Daily-Exploration

Recursive Self-Improvement through daily optimization experiments. Each day explores a new three-layer self-restructuring optimizer combining ideas from meta-learning, quality-diversity, and cross-domain algorithm transfer.

## Project Structure

```
RSI-Daily-Exploration/
├── requirements.txt          # Shared dependencies (numpy, scipy)
├── explorations/
│   ├── 0327/                 # Mar 27, 2026
│   │   ├── README.md         # DGM-MAP-SA documentation
│   │   └── meta_transfer.py  # DGM-MAP-SA implementation
│   └── 0329/                 # Mar 29, 2026
│       ├── README.md         # CMAES-MAESTRO-ALMA documentation
│       └── meta_transfer.py  # CMAES-MAESTRO-ALMA implementation
```

## Explorations

| Date | Name | Layer 1 | Layer 2 | Layer 3 |
|------|------|---------|---------|---------|
| 0327 | DGM-MAP-SA | MAP-Elites (grid archive) | Heuristic meta-restructurer | SA + PSO injection |
| 0329 | CMAES-MAESTRO-ALMA | CMA-ES (learned covariance) | MAESTRO contextual bandit | DE + ALMA memory search |

## Quick Start

```bash
pip install -r requirements.txt

# Run a specific day's exploration
python explorations/0327/meta_transfer.py
python explorations/0329/meta_transfer.py
```

## Architecture

Each exploration implements a three-layer self-restructuring optimizer:

**Layer 1 (Optimizer):** Solves the optimization problem directly.

**Layer 2 (Meta-optimizer):** Watches Layer 1 for stagnation and restructures the optimization itself — modifying loss functions, changing topology, or re-weighting objectives.

**Layer 3 (Expansion monitor):** Monitors whether Layer 2 is genuinely expanding the reachable solution space or merely rearranging existing solutions. When expansion stalls, it injects structurally different algorithm components from other metaheuristic families.

## License

This project is for research and educational purposes.
