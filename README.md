# STL Games: A Toolbox for Control Synthesis from Signal Temporal Logic Tasks

## 🔍 Overview

**STL Games** is a Python toolbox for synthesizing control strategies for multi-agent systems that satisfy Signal Temporal Logic (STL) tasks. It focuses on reducing computational time while ensuring robustness by integrating Control Barrier Functions (CBFs) and time-varying sets into a model predictive control (MPC) framework.  

The control synthesis is formulated within a differential game framework, where each robot is treated as an independent decision-making agent. In this setting, each robot solves has a distinct cost function and set of constraints in a centrally solved MPC, reflecting its specific STL task and timing constraints. 

---

## 📦 Features

- Define STL specifications and system dynamics
- Construct controllers using MPC + CBFs + time-varying sets to reduce solve time
- Visualize trajectories and compute robustness
- Support for multi-agent differential game formulation

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

### sim1.py
A control strategy is synthesized for a configurable number of robots, each governed by double integrator dynamics. The simulation takes place in a continuous 2D environment. The high-level behavior of each robot is specified using STL (Signal Temporal Logic) tasks, including:
- Obstacle avoidance: Robots must remain outside fixed circular obstacle regions.
- Goal reaching: Robots must reach one of several available square goal areas within a specified time window.  


Each robot is assigned a set of possible goals and must choose one to satisfy the STL specification. STL formula is in the form:  
![STL Formula](media/sim1-STLFragment.png)

The output includes visualizations (robot trajectories, goal areas, obstacles), execution time for the control synthesis and robustness metrics for the trajectory. The results are typically generated in less than 5 seconds.

### sim2.py
This simulation builds upon sim1.py by introducing a significantly more challenging obstacle scenario. Obstacles remain circular but are both larger and more numerous, creating narrow passages that complicate the path-planning and control synthesis. Unlike the first simulation, each robot may have a unique time deadline for satisfying its STL task, adding an additional layer of complexity to the multi-agent coordination problem. The unique deadlines imply that the agents need to cooperate or avoid conflicts with different urgency levels. Below is a visual representation of the second simulation scenario.

![sim2 scenario](media/sim2-scenario.png)