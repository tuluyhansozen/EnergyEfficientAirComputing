# AirCompSim - Energy-Efficient Air Computing Simulation

## Overview
**AirCompSim** is a discrete event simulator designed to evaluate **air computing environments**, integrating **UAVs, edge servers, and cloud computing** for **dynamic task offloading**. 

### Current Limitations:
- **No energy efficiency calculations** (UAVs and edge servers do not track power usage).
- **No energy-aware scheduling** (tasks are assigned based on availability, not efficiency).
- **Static UAV behavior** (UAVs do not adapt based on power constraints).

## Goal: Energy Efficiency Enhancement
We aim to extend **AirCompSim** by integrating **energy-aware models** for **power-efficient computation and resource allocation**.

### Planned Enhancements:
1. **UAV Energy Models**
   - Implement UAV **battery consumption** based on **task load, flight distance, and idle states**.
   - Simulate **battery depletion & recharging mechanisms**.

2. **Energy-Aware Task Scheduling**
   - Develop **adaptive offloading strategies** that prioritize **energy-efficient task placement**.
   - Balance **UAV offloading vs. edge server usage** for **minimum power consumption**.
