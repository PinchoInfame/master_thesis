# STL Games: A Toolbox for Control Synthesis from Signal Temporal Logic Tasks

## 🔍 Overview

**STL Games** is a Python toolbox for synthesizing control strategies for multi-agent systems that satisfy Signal Temporal Logic (STL) tasks. It focuses on reducing computational time while ensuring robustness by integrating Control Barrier Functions (CBFs) and time-varying sets into a model predictive control (MPC) framework.

---

## 📦 Features

- Define STL specifications and system dynamics
- Construct controllers using MPC + CBFs
- Use CBF and time-varying sets to reduce solve time
- Visualize trajectories and compute robustness

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
A control is sinthetized for a varying number of robots. The high-level behavior of each robot is specified using STL tasks. STL tasks include:
- Obstacle avoidance: Robots must stay outside fixed circular obstacle regions.
- Goal reaching: Robots must reach one of several available square goal areas within a specified time window. 
Each robot is assigned a set of possible goals and must choose one to satisfy the STL specification. STL formula is in the form:  
![STL Formula](media/sim1-STLFragment.png)
