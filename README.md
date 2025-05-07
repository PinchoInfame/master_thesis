# STL Games: A Toolbox for Control Synthesis from Signal Temporal Logic Tasks

## 🔍 Overview

**STL Games** is a Python toolbox for synthesizing control strategies that satisfy Signal Temporal Logic (STL) tasks. It focuses on reducing computational time while ensuring robustness by integrating Control Barrier Functions (CBFs) and time-varying sets into a model predictive control (MPC) framework.

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