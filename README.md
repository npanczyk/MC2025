# Post Hoc Explainable AI for Nuclear 
## Alex Xu and Nataly Panczyk

## Overview
This repository contains a set of scripts used to generate results for "Towards Explainable AI in Nuclear: Introducing Ad Hoc Model Explainability" by Xu et al. presented at the 2025 ANS Math and Comp Conference in Denver, CO. The scripts are designed to compare feature attribution scores and methods for a random forest and a feedforward neural network trained on a critical heat flux dataset, and the attached paper details methods and results. This repository is intended for those seeking to reproduce our results. NOTE: Values produced using files in this repo will vary slightly from those presented in the conference paper. The data used in the paper are proprietary, so synthetic data are provided in this repo for demonstration purposes.

---

## Contents

- **Scripts**
  - `helpers.py`: Helper functions like data preprocessing and metric calculations.
  - `rf.py`: Generates a random forest and feature importance plots.
  - `nn_explain`: Generates a feedforward neural network with best hyperparameters according to pyMAISE for the CHF dataset, then conducts SHAP analysis and plotting.
  - `correlation.py`: Generates a correlation matrix for the CHF synthetic dataset.

- **Conference Paper**
  - `conference_paper.pdf`: "Towards Explainable AI in Nuclear: Introducing Ad Hoc Model Explainability" by Alex Xu and Nataly Panczyk, presented at the 2025 American Nuclear Society's Math and Computation Conference

---

## Getting Started

### Prerequisites

Before running the scripts, create a new environment using the packages using `environment.yml`.

## Running the Scripts
Simply run `rf.py`, `nn_explain.py`, and `correlation.py` to reproduce all results.

