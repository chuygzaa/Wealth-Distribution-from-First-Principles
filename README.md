# Wealth Distribution from First Principles: An Agent-Based Simulation of Merit, Luck and Dynastic Survival

**An Agent-Based Simulation of Merit, Luck, and Dynastic Survival**

This repository contains the Python-based operations research model and interactive dashboard that powers the working paper: *[Insert SSRN Link Here: Wealth Distribution from First Principles]*.

Instead of relying on top-down macroeconomic equilibrium equations, this engine treats the economy as a closed thermodynamic system governed by accounting conservation laws, compounding mathematics ($r > g$), and localized stochastic exchange. 

You can interact with the live simulation via the Streamlit Community Cloud here: **https://wealth-distribution-from-first-principles-ebswavymq725e8ym6bmf.streamlit.app/**

---

## 🧠 Project Overview

Mainstream macroeconomic models frequently rely on representative agents and equilibrium assumptions. This project utilizes a "Business Physics" methodology, building an economic system from the bottom-up. 

By running the simulation across 1,000 rounds (roughly 12–13 overlapping generations) and sweeping an estate tax from 0% to 100%, the engine maps:
1. **The Inevitability of Oligarchy:** How wealth-weighted advantage guarantees a Gini coefficient of ~0.93 absent intervention.
2. **The Meritocracy Paradox:** A U-curve proving that an intermediate estate tax (40%-50%) maximizes the alignment of intrinsic skill with terminal wealth. 
3. **The Liquidation Trap:** How baseline consumption friction ($C_i > L_i$) forces the bottom 50% to sell compounding assets, creating a structural upward transfer of wealth.
4. **Founder's Luck:** The centuries-long, path-dependent echo of a Generation 1 ancestor's initial stochastic outcomes.

---

## 📂 The Evolutionary File Structure

To maintain transparency and allow others to audit the underlying mechanics, the code is preserved in 7 distinct evolutionary iterations. This allows researchers to see exactly how the introduction of each new constraint alters the macroscopic outcomes.

* **`Iteration 1 - Two-Agent Pairwise Betting and Parameter Sweeps.py`** The initial proof-of-concept pitting two agents in a zero-sum game to test the baseline math of proportional betting and system growth.
* **`Iteration 2 - 1000 Agents Pure Stochastic Exchange.py`** Expands the system to 1,000 agents to track the Gini coefficient and early "Origin Stories" using a purely fair (50/50) stochastic exchange.
* **`Iteration 2.5 - Wealth-Weighted Advantage without Skill.py`** Introduces Proportional Odds, proving that capital scale alone ("wealth-attained advantage") inevitably creates an oligarchy even when human skill is non-existent.
* **`Iteration 3 - Skill Multiplier and Wealth-Weighted Advantage.py`** Introduces a Gaussian distribution of intrinsic human talent (Skill), transforming the exchange into a "Merit vs. Capital" battleground.
* **`Iteration 4 - Labor Wages and Consumption Friction.py`** Introduces the three-step macroeconomic cycle: heterogeneous wages, the cost-of-living consumption friction, and corporate dividend recycling.
* **`Iteration 5 - Overlapping Generations and Estate Taxation.py`** Introduces "deep time" via 80-round lifespans, death, noisy skill inheritance, and the primary independent variable: the Estate Tax rate.
* **`Iteration 6 - Dual-Asset Liquidation Trap and Streamlit Dashboard.py`** The master physics engine. Bifurcates wealth into liquid wages and appreciating assets to operationalize $r > g$, enforces the "Liquidation Trap," and wraps the simulation in a multi-page interactive Streamlit UI.

---

## 🚀 How to Run Locally

To run the final interactive dashboard (Iteration 6) on your local machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/Inequality-Simulator.git](https://github.com/yourusername/Inequality-Simulator.git)
   cd Inequality-Simulator
   
2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate #on Windows use 'venv\Scripts\activate
   
3. **Install the required dependencies:**
   ```bash
   pip install numpy pandas matplotlib streamlit
   
4. **Launch the Streamlit app:**
   ```bash
   streamlit run "Iteration 6 - Dual-Asset Liquidation Trap and Streamlit Dashboard.py"
   

⚖️ Author's Disclaimer & IP Notice
Author: Jesus Alejandro Garza López

Date: April 2026

The views, models, and conclusions presented in this repository and its accompanying paper are strictly those of the author's. This research was conducted entirely independently, utilizing personal resources, and does not draw upon or relate to any proprietary data or models.

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.