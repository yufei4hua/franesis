# Franka Robot Arm Control Based on Genesis

## Overview

This is a course project for ARCL (Advanced Robot Control and Learning). The task is decoupled force and motion control.

This repository implemented a simple gym style environment of franka emika panda simulated with Genesis. 

The controllers included methods from paper "Hybrid Force-Impedance Control for Fast End-Effector Motions" and "Partially Decoupled Impedance Motion Force Control Using Prioritized Inertia Shaping".

## Installation

### Prerequisites

Install [Pixi](https://pixi.sh) if you don't have it already.

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

### One-line installation & activation

```bash
pixi shell
```
## Controllers
### Cartesian impedance controller
```bash
python scripts/sim.py -e default -c imp
```
### Force Motion Controller
```bash
python scripts/sim.py -e box -c mfc
```
### Hybrid Force Impedance Controller
```bash
python scripts/sim.py -e box -c hfic
```
### Partial Decoupled Controller
```bash
python scripts/sim.py -e box -c pdimfc
```


