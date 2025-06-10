# STL Games: A Toolbox for Control Synthesis from Signal Temporal Logic Tasks

## 🔍 Overview

**STL Games** is a Python toolbox for synthesizing control strategies for multi-agent systems that satisfy Signal Temporal Logic (STL) tasks. It integrates time-varying sets  obtained from Control Barrier Functions (CBFs) into a model predictive control (MPC) framework. The focus is on reducing computational time while ensuring robustness to STL specifications.

We leverage a hybrid approach that integrates centralized and decentralized components. Individual STL tasks that do not require coordination are solved with a centralized method to obtain a high-level planner. Then, conflicts in shared STL tasks are solved iteratively in a decentralized step, leveraging a game-theoretic framework. We define STL games, an extension of differential games, where the state is constrained to satisfy STL specifications.
 

---

## 📦 Features

- Construct controllers using MPC + CBFs + time-varying sets to reduce solve time
- Handle disjunctions with a limited number of Boolean variables
- set-up and solve STL games
- Visualize trajectories and compute robustness
- Support for distributed multi-agent formulation and STL tasks that require coordination among agents

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/PinchoInfame/master_thesis.git
cd master_thesis
pip install -r requirements.txt
pip install -e . 
```

### 2. Gurobi license
This project relies on the Gurobi Optimizer to solve optimization problems involved in control synthesis. To run simulations successfully, you must install and activate a valid Gurobi license. Academic users can request a free license here:
https://www.gurobi.com/academia/academic-program-and-licenses/

---

## Simulation Results
All simulations take place in a continuous 2D environment with a varying number of agents, each governed by double integrator dynamics. The high level behaviour desired for the multi-robot problem is defined by STL specifications.
### sim1.py
A control strategy is synthesized for a configurable number of agents. Each agent is requested to:
- Avoid obstacles: agents must remain outside fixed circular obstacle regions.
- Avoid collision with other agents
- Reach a goal: each agent must reach one of its several designated square goal areas within a specified time window. 


Each agent is assigned a set of possible goals (disjunction) and can have an arbitrary time deadline to reach one of them. These high-level tasks are specified using STL grammar.
The problem is solved centrally using a state-of-the-art MILP solver (available with stlpy).

The output includes visualizations (trajectories, goal areas, obstacles), execution time for the control synthesis and robustness metrics for the trajectory. 

### sim2.py
The simulation scenario is the same as sim1.py. In this case the control synthesis is solved using the hybrid method. A high-level planner is generated with a centralized MPC enforcing STL requirements for goal reaching and obstacle avoidance.
Then, collision between agents are solved iteratively utilizing a partial-information STL game formulation.  

Both steps rely on CBF-based time-varying sets within an MPC formulation. A limited number of Boolean variables is sufficient to handle disjunctions.

Number of agents, number of goals, time windows can be easily modified to compare performance of the two approaches (eg: scalability).  

This approach balances computational efficiency with robustness to conflicts, making it scalable and modular. The simulation output includes the final, collision-free trajectories, visual comparisons with the initial plan, and execution statistics across the two phases.
