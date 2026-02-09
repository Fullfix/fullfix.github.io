---
layout: post
title: "Introducing My Pruning Library"
date: 2026-02-09
categories: notes
---


LLM pruning research is often hindered by the engineering complexity of reproducing activation-aware methods, which usually require custom hooks and intricate layer-wise management. To lower the barrier for experimentation, I developed [nn-pruning](https://github.com/Fullfix/nn-pruning): a modular PyTorch toolkit that standardizes activation collection and benchmarking. By decoupling pruning logic from the underlying model infrastructure, the project allows researchers to implement and compare new algorithms like Wanda or SparseGPT with minimal boilerplate.

<!--more-->

**Currently supported**:
* Sparsity Patterns: Unstructured and Semi-structured `N:M`
* Model Families: OPT (`facebook/opt`)

**Repository** [nn-pruning](https://github.com/Fullfix/nn-pruning)

To validate the toolkit, I reproduced the benchmarks for the OPT model family across three different sparsities: Unstructured (50%), Semi-structured 2:4, and 4:8.

**WikiText-2 Perplexity Results**
(Calibration: 128 C4 sequences, 2048 tokens each. Sparsity applies to Attention and MLP linear weights.)

| Method | Sparsity | 125M | 350M | 1.3B   | 2.7B   | 6.7B | 13B    |
| :--- | :--- | :--- | :--- |:-------|:-------| :--- |:-------|
| **Dense** | 0% | 27.65 | 22.02 | 14.63  | 12.46  | 10.86 | 10.13  |
| | | | |        |        | |        |
| **Magnitude** | 50% | 197.38 | 97.11 | 1.6e3  | 255.16 | 959.48 | 1.2e4  |
| **Wanda** | 50% | 38.78 | 36.52 | 18.61  | 14.46  | 11.88 | 12.04  |
| **SparseGPT** | 50% | 38.31 | 32.31 | 17.97  | 13.77  | 11.71 | 11.14  |
| | | | |        |        | |        |
| **Magnitude** | 2:4 | 347.51 | 416.56 | 444.39 | 1.1e3  | 265.80 | 468.95 |
| **Wanda** | 2:4 | 78.80 | 107.12 | 27.29  | 21.84  | 15.91 | 16.51  |
| **SparseGPT** | 2:4 | 63.69 | 56.36 | 24.18  | 16.87  | 13.83 | 12.96  |
| | | | |        |        | |        |
| **Magnitude** | 4:8 | 171.28 | 160.52 | 256.32 | 155.48 | 214.14 | 459.81 |
| **Wanda** | 4:8 | 51.91 | 58.17 | 21.88  | 17.04  | 13.42 | 13.94  |
| **SparseGPT** | 4:8 | 46.91 | 40.20 | 20.18  | 14.80  | 12.53 | 11.86  |
