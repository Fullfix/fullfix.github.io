---
layout: post
title: "FROG: My attempt to create efficient second-order optimizer"
date: 2026-01-25
categories: notes
---

FROG (<u>F</u>isher <u>RO</u>w-wise Preconditionin<u>G</u>) is a second-order optimizer based on row-wise Fisher preconditioning. It uses joint Conjugate Gradient solves to approximate natural-gradient updates with low computational overhead. Fisher trace–based normalization ensures scale-free updates. The method is applicable to linear and convolutional layers and requires only a small number of CG iterations in practice. Implementation is available at [GitHub](https://github.com/Fullfix/frog-optimizer).

<!--more-->

**Download:** [frog-technical-overview.pdf](/assets/pdfs/frog-technical-overview.pdf)

**Technical Overview**

<iframe
  src="/assets/pdfs/frog-technical-overview.pdf#pagemode=none&navpanes=0&toolbar=0"
  width="100%"
  height="900"
  style="border: 1px solid rgba(0,0,0,0.15); border-radius: 8px;"
></iframe>