# Multi-Colony Agent-Based Simulation

This repository contains an agent-based simulation framework for studying
multi-colony population dynamics in a competitive resource-constrained
environment.

The project compares three algorithmic strategies:
- Greedy rule-based heuristics
- Decentralized Q-learning
- A colony-fitness-guided Particle Swarm Optimization (PSO) variant

## Overview
Multiple colonies of agents compete on a 2D grid world by foraging resources,
engaging in combat, reproducing, and expanding territory. Each colony is
homogeneous and uses a single learning paradigm, allowing controlled
comparison of emergent behaviors.

## Algorithms Implemented
- **Greedy Heuristic**: Immediate reward maximization based on local features
- **Q-Learning**: Independent agent learning with local state representation
- **PSO Variant**: Swarm-inspired coordination guided by colony-level fitness

## Key Concepts
- Multi-agent systems
- Reinforcement learning
- Swarm intelligence
- Emergent behavior
- Agent-based modeling

## Technologies
- Python
- NumPy
- Pygame (for visualization)
- Matplotlib (for analysis)

## Report
The full project report with problem formulation, algorithms, experiments,
and results is available in the `report/` directory.

## Authors
- Hamna Sajid 
- Syed Taha
- Ammar Khan
