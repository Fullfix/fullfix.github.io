---
layout: post
title: "Properties of the Fisher Information Matrix"
date: 2026-04-18
categories: notes
---

This post introduces the Fisher Information Matrix and develops its main statistical and geometric properties. It concludes with a short discussion of what Fisher singularity means and how it arises in overparameterized models.

<!--more-->

## Statistical Interpretation

We start by introducing Fisher Information Matrix and describing its statistical properties which help understand the intuition behind the word "information" in the name.

First, let us define the set of distributions that we will be working with. We understand distribution $p_{\theta}: \mathbb{R}^{d} \to \mathbb{R}$ as a map such that $\int p_{\theta}(x)\,dx = 1$ and $p_{\theta}(x) \ge 0$ for all $x$. Each $\theta \in \Theta \subset \mathbb{R}^{n}$ defines its own distribution and is called parameter. The open set $\Theta$ is called a parameter space. The set of distributions that are formed by all possible parameters is called a statistical model with dimension $n$:

$$
S := \{\, p_\theta \mid \theta \in \Theta \,\}.
$$

Strictly speaking, the mapping $\theta \leftrightarrow p_\theta$ should be bijective and $\Theta$ should be an open set in $\mathbb{R}^n$ for the correct definition of the statistical model. However, due to the common case of overparameterization in neural networks, we omit this requirement in the section on singularity, where we analyze the connection between overparameterization and Fisher matrix.

We will make the following assumptions about the statistical model and the distributions:

$$
\forall x\in\mathbb X,\quad \theta\mapsto p_\theta(x)\in C^\infty(\Theta)
$$

$$
\operatorname{supp}(p_\theta)=\mathbb X \quad \text{for all } \theta\in\Theta
$$

$$
\text{Regularity for } \frac{\partial }{\partial \theta }p_\theta(x),\ s_\theta(x),\ s_\theta(x)s_\theta(x)^T,\ \frac{\partial^2}{\partial\theta^2}\log p_{\theta}(x)
$$

$$
\mathbb{E}_\theta \left[\left|s_\theta s_\theta^T\right|\right] < \infty
$$

Here we denote

$$
s_\theta(x) := \frac{\partial}{\partial \theta}\log p_\theta(x)
$$

and call it a **score function**.

The first assumption is classical for statistical models. We typically require sufficiently smooth dependence on $\theta$ for derivatives to exist.

The second assumption implies that the support of the distribution does not depend on $\theta$. It is an important requirement for interchanging $\frac{\partial}{\partial \theta}$ and $\int$ and filters out distributions such as Uniform $U[0, \theta]$.

The third assumption requires existence of the (element-wise) integrals and ability to interchange $\frac{\partial}{\partial \theta}$ and $\int$. More strictly, for a function
$f_\theta(x) \in \{ \frac{\partial }{\partial \theta }p_\theta(x), \ldots \}$ we need:

$$
\begin{aligned}
\forall \theta_0\in\Theta,\ &\exists \text{ a neighborhood } U \ni \theta_0,\ \exists g\in L^1(\mathbb X) \\
&\text{such that }
|f_\theta(x)|\le g(x)
\quad \forall \theta\in U,\ \forall x\in\mathbb X.
\end{aligned}
$$

The fourth assumption is required for the existence of the Fisher Information Matrix, which will be introduced later.

The score function $s_\theta(x)$ plays an important role in machine learning. Using chain rule, we can express density derivative in terms of the score:

$$
s_\theta(x)=\frac{\partial}{\partial \theta}\log p_\theta(x)
= \frac{1}{p_\theta(x)}\frac{\partial}{\partial \theta}p_\theta(x)
\implies
\frac{\partial}{\partial \theta}p_\theta(x) = s_\theta(x)p_\theta(x)
$$

The main property of the score function is having zero mean:

$$
\mathbb{E}_\theta [s_\theta]
=
\int s_\theta(x)p_\theta(x)\,dx
=
\int \frac{\partial}{\partial \theta}p_\theta(x)\,dx
\overset{\text{regularity}}{=}
\frac{\partial}{\partial \theta} \int p_\theta(x)\,dx
=
\frac{\partial}{\partial \theta} 1
=
0
$$

Let us define the Fisher Information Matrix, the main object of the current document. We will also call it Fisher matrix or just Fisher.

$$
F(\theta) := \mathbb{E}_\theta \left[s_\theta s_\theta^T\right] = \operatorname{Cov}(s_\theta) \succeq 0
$$

Fisher matrix is also the covariance matrix of the score function due to zero mean of the latter. By definition, it is positive semidefinite.

<div class="framed" markdown="1">

<div class="statementinner" markdown="1">

**Statement.**

$$
F(\theta)
=
\mathbb{E}_\theta\left[s_\theta s_\theta ^T\right]
=
\mathbb{E}_\theta \left[\frac{\partial^2}{\partial \theta^2}(-\log p_\theta )\right]
$$

</div>

</div>

#### Proof

Using the score property:

$$
\mathbb{E}_\theta \left[s_\theta \right] = \int s_\theta(x)p_\theta(x)\,dx = 0
$$

Differentiating by $\theta$ on both sides (and using regularity) gives:

$$
\int \left[
\left(\frac{\partial}{\partial \theta}s_\theta(x)\right) p_\theta(x)
+
s_\theta(x)\left(\frac{\partial}{\partial \theta}p_\theta(x)\right)^T
\right]dx = 0
$$

Rewriting the score by the definition on the left side and substituting the identity
$\frac{\partial}{\partial \theta}p_\theta(x) = s_\theta(x)p_\theta(x)$
on the right side proves the statement.

---

Next, we would like to discuss the connection of the Fisher matrix with sufficient statistics. By statistic we mean any smooth function $T: \mathbb{X} \to \mathbb{R}^m$. We assume sufficient smoothness $T(x) \in C^{\infty}(\mathbb{X})$. If we denote $X$ as a random variable with distribution $p_\theta(x)$, then we denote $T(X)$ as a random variable with distribution:

$$
q_\theta (t)
=
\int \delta (T(x) - t) p_\theta(x)\,dx
=
\mathbb{E}_\theta \bigl[\delta (T(x) - t)\bigr]
$$

We call $T(X)$ a sufficient statistic if $p_\theta (x) / q_\theta (T(x))$ does not depend on $\theta$, or in other words $p_\theta(x) = c(x)q_\theta(T(x))$. In fact, it is sufficient to find any positive function $g_\theta$, such that $p_\theta(x) = c(x)g_\theta(T(x))$. This equivalence is called factorization theorem. It can be easily proved with the following observation:

$$
q_\theta (t)
=
\int \delta(T(x) - t)\, c(x)g_\theta(T(x))\,dx
=
g_\theta(t)C(t)
$$

So we can rewrite factorization with any function $g_\theta$ as follows:

$$
p_\theta(x)
=
c(x)g_\theta(T(x))
=
\left(\frac{c(x)}{C(T(x))}\right)q_\theta(T(x))
$$

<div class="framed" markdown="1">

<div class="statementinner" markdown="1">

**Statement.**

If $T(x)$ is a statistic for the statistical model $\{ p_\theta \}$ and we denote

$$
F(\theta) = \mathbb{E}_\theta \left[ s_\theta s_\theta^T \right],
\qquad
s_\theta(x) = \frac{\partial}{\partial \theta} \log p_\theta(x)
$$

$$
F_T(\theta) = \widetilde{\mathbb{E}}_\theta \left[ \widetilde{s}_\theta \widetilde{s}_\theta^T \right],
\qquad
\widetilde{s}_\theta(t) = \frac{\partial}{\partial \theta} \log q_\theta(t)
$$

then $F(\theta) \succeq F_T(\theta)$, and equality holds if and only if $T$ is a sufficient statistic.

</div>

</div>

#### Proof

For any value $t$ of the statistic:

$$
q_\theta(t) = \int \delta (T(x) - t )p_\theta(x)\,dx
$$

Differentiate with respect to $\theta$:

$$
\frac{\partial}{\partial \theta} q_\theta(t)
=
\int \delta(T(x) - t) p_\theta(x)s_\theta(x)\,dx
$$

Now, dividing both sides by $q_\theta(t)$ and using the definition of conditional expectation, we get:

$$
\widetilde{s}_\theta(t) = \mathbb{E}_\theta\left[s_\theta \mid T = t \right]
$$

For any fixed $t$:

$$
\operatorname{Cov}(s_\theta \mid T=t)
:=
\mathbb{E}_\theta[s_\theta s_\theta^T \mid T = t]
-
\mathbb{E}_\theta[s_\theta \mid T = t]\mathbb{E}_\theta[s_\theta \mid T = t]^T
\succeq 0
$$

Now we take expectation with respect to $T$:

$$
\begin{aligned}
\widetilde{\mathbb{E}}\left[\mathbb{E}_\theta\left[s_\theta s_\theta^T \mid T \right] \right]
&=
\int
\frac{
\int \delta(T(x)-t)\, s_\theta(x)s_\theta(x)^T p_\theta(x)\,dx
}{
q_\theta(t)
}
q_\theta(t)\,dt
\\
&=
\int
s_\theta(x)s_\theta(x)^T p_\theta(x)
\left(
\int \delta(T(x)-t)\,dt
\right) dx
\\
&=
\int s_\theta(x)s_\theta(x)^T p_\theta(x)\,dx
\\
&=
\mathbb{E}_\theta\left[s_\theta s_\theta^T\right]
\end{aligned}
$$

After integrating with positive factor $q_\theta(t)$, the covariance matrix still remains positive semidefinite. We also substitute $\widetilde{s}_\theta(t)$ for the second term:

$$
\mathbb{E}_\theta \left[s_\theta s_\theta^T \right]
-
\widetilde{\mathbb{E}}_\theta\left[\widetilde{s}_\theta \widetilde{s}_\theta^T \right]
\succeq 0
$$

Hence

$$
F(\theta) - F_T(\theta) \succeq 0
$$

Now note that equality to zero means that the expectation of a positive semidefinite matrix is zero, thus equivalently for all $t$:

$$
\operatorname{Cov}(s_\theta \mid T = t) = 0
$$

Equivalently, after fixing $T = t$, $s_\theta(x)$ becomes a constant. We can express this as:

$$
s_\theta(x)
=
\frac{\partial}{\partial \theta}\log p_\theta(x)
=
a_\theta(T(x))
$$

For fixed $t$ and $x$ with $T(x) = t$, we can differentiate $a_\theta$ to obtain:

$$
\frac{\partial}{\partial \theta_j} (a_\theta(T(x)))_i
=
\frac{\partial^2}{\partial \theta_i \partial \theta_j}\log p_\theta(x)
=
\frac{\partial}{\partial \theta_i}(a_\theta(T(x)))_j
$$

So $a_\theta$ can be expressed as a gradient of some scalar function. Reparameterize:

$$
a_\theta(t) = \frac{\partial}{\partial \theta}\log g_\theta(t)
$$

Then

$$
\frac{\partial}{\partial \theta}\left[\log p_\theta(x) - \log g_\theta(T(x)) \right]
=
\frac{\partial}{\partial\theta}\log \frac{p_\theta(x)}{g_\theta(T(x))}
=
0
$$

This results in $p_\theta(x) = c(x) g_\theta(T(x))$ and completes the proof by the factorization theorem.

---

This theorem shows us an intuitive interpretation of Fisher matrix: when performing any transformation $T(x)$, we lose “information” about the parameter $\theta$. However, when $T(x)$ is a sufficient statistic, it keeps all “information” about $\theta$.

We call a statistic $T: \mathbb{X} \to \mathbb{R}^n$ **locally unbiased** at point $\theta_0$ if for any $\theta$ from some neighborhood of $\theta_0$ we have:

$$
\mathbb{E}_\theta \left[T\right] = \theta
$$

Now comes the famous Cramér–Rao inequality which provides a lower bound on the estimation error for an unbiased estimator.

<div class="framed" markdown="1">

<div class="statementinner" markdown="1">

**Statement (Cramér–Rao inequality).**

Let $\{p_\theta \}$ be a statistical model and $T: \mathbb{X} \to \mathbb{R}^n$ be a locally unbiased estimator for some parameter $\theta \in \mathbb{R}^n$. Denote:

$$
F(\theta) = \mathbb{E}_\theta\left[ s_\theta s_\theta^T \right] \succ 0,
\qquad
s_\theta(x) = \frac{\partial}{\partial \theta}\log p_\theta(x)
$$

Then:

$$
\operatorname{Cov}_\theta(T) \succeq F(\theta)^{-1}
$$

</div>

</div>

#### Proof

It is easy to show that for a locally unbiased estimator which satisfies
$\mathbb{E}_{\widetilde{\theta}}\left[T\right] = \widetilde{\theta}$
for some neighborhood of $\theta$:

$$
\frac{\partial}{\partial \theta} \mathbb{E}_\theta \left[T\right]
=
\mathbb{E}_\theta \left[ Ts_\theta \right]
=
I_n
$$

Using the score function property $\mathbb{E}_\theta s_\theta = 0$, we can subtract
$0 = \theta \mathbb{E}_\theta s_\theta$:

$$
\mathbb{E}_\theta \left[ (T - \theta)s_{\theta}\right]
=
I_n
=
\operatorname{Cov}(T - \theta, s_\theta )
$$

If we consider the stacked random vector
$\begin{bmatrix} T - \theta \\ s_\theta \end{bmatrix}$
and take its covariance matrix, we get:

$$
\begin{bmatrix}
\operatorname{Cov}(T) & I_n \\
I_n & F(\theta)
\end{bmatrix} \succeq 0
$$

Schur complement with respect to $F(\theta)$ gives:

$$
\operatorname{Cov}(T) - I_n F(\theta)^{-1}I_n \succeq 0
$$

which proves the theorem.

---

Cramér–Rao inequality allows us to interpret Fisher matrix as a lower bound on the uncertainty of estimation. If $F(\theta)$ has eigenvalues close to $0$, its inverse has large eigenvalues and therefore estimation is very poor. On the contrary, large eigenvalues of $F(\theta)$ provide a lot of “information” about the parameter for the estimator. The estimators that achieve equality in Cramér–Rao are called efficient.

Next property allows us to understand how Fisher behaves when distribution consists of independent variables and is factorized by them.

<div class="framed" markdown="1">

<div class="statementinner" markdown="1">

**Statement.**

Let $X_1, X_2$ be independent random variables with distributions $p_{\theta_1}(x), p_{\theta_2}(x)$. Denote $\theta = [\theta_1, \theta_2]$ as stacked parameter. Denote $F_1(\theta_1), F_2(\theta_2)$ as Fisher matrices of individual distributions, and $F(\theta)$ as Fisher matrix of the joint distribution
$p_\theta(x) = p_{\theta_1}(x_1)p_{\theta_2}(x_2)$.
Then joint Fisher matrix is block-diagonal:

$$
F(\theta) =
\begin{bmatrix}
F_1(\theta_1) & 0 \\
0 & F_2(\theta_2)
\end{bmatrix}
$$

</div>

</div>

#### Proof

When considering full parameter $\theta$, we have:

$$
s_{\theta}(x)
=
\frac{\partial}{\partial \theta}\log \bigl(p_{\theta_1}(x_1)p_{\theta_2}(x_2)\bigr)
=
\begin{bmatrix}
s_{\theta_1}(x_1) \\
0
\end{bmatrix}
+
\begin{bmatrix}
0 \\
s_{\theta_2}(x_2)
\end{bmatrix}
=
\begin{bmatrix}
s_{\theta_1}(x_1) \\
s_{\theta_2}(x_2)
\end{bmatrix}
$$

Now, Fisher matrix has form:

$$
F(\theta)
=
\mathbb{E}_\theta
\begin{bmatrix}
s_{\theta_1}s_{\theta_1}^T & s_{\theta_1} s_{\theta_2}^T \\
s_{\theta_2}s_{\theta_1}^T & s_{\theta_2}s_{\theta_2}^T
\end{bmatrix}
=
\begin{bmatrix}
\mathbb{E}_\theta \left[ s_{\theta_1}s_{\theta_1}^T \right] & 0 \\
0 & \mathbb{E}_\theta \left[ s_{\theta_2}s_{\theta_2}^T \right]
\end{bmatrix}
=
\begin{bmatrix}
F_1(\theta_1) & 0 \\
0 & F_2(\theta_2)
\end{bmatrix}
$$

where off-block-diagonal entries vanish due to expectation factorization and the zero-mean property of the score function.

---

So, independence of the parameters is nicely reflected in the Fisher matrix. The inverse implication, however, is generally false: block-diagonal Fisher matrix does not imply independence. Next, when all independent distributions depend on the same parameter, we also get a specific form of the joint Fisher matrix.

<div class="framed" markdown="1">

<div class="statementinner" markdown="1">

**Statement.**

Let $X_1, \ldots, X_N$ be $N$ independent random variables with distributions
$p_{\eta_1}(x), \ldots, p_{\eta_N}(x)$, where $\eta_i = f_i(\theta)$ and $\theta$ is the shared parameter.
The joint distribution has form:

$$
p_\theta(x_1, \ldots, x_N) = p_{\eta_1(\theta)}(x_1) \cdots p_{\eta_N(\theta)}(x_N)
$$

Let us denote $F_1(\theta), \ldots, F_N(\theta)$ as Fisher matrices of individual distributions, and $F(\theta)$ as Fisher of the joint distribution. Then:

$$
F(\theta) = F_1(\theta) + \cdots + F_N(\theta)
$$

</div>

</div>

#### Proof

Score function of the joint distribution takes form:

$$
s(x_1, \ldots, x_N)
=
\frac{\partial}{\partial \theta}\log p_\theta(x_1, \ldots, x_N)
=
\frac{\partial}{\partial \theta} \sum_{i=1}^N \log p_{\eta_i(\theta)}(x_i)
=
\sum_{i=1}^N s_{\theta}^{(i)}(x_i)
$$

where

$$
s_\theta^{(i)}(x_i) := \frac{\partial}{\partial \theta}\log p_{\eta_i(\theta)}(x_i)
$$

Then:

$$
F(\theta)
=
\mathbb{E}_{\theta}\left[ss^T\right]
=
\sum_{i=1}^N \mathbb{E}_{\theta}\left[ s_{\theta}^{(i)}\left(s_{\theta}^{(i)}\right)^T \right]
+
\sum_{i \ne j} \mathbb{E}_{\theta}\left[s_{\theta}^{(i)}\right]\mathbb{E}_{\theta}\left[s_{\theta}^{(j)} \right]^T
=
\sum_{i=1}^N F_{i}(\theta)
$$

where the second term with $i \ne j$ vanishes due to the zero-mean property of the score function.

---

Intuitively, this means that when independent variables are combined, their “information” is added.

When considering estimation with $N$ independent samples from distribution $p_{\theta}$ with Fisher matrix $F(\theta)$, we get the simplification:

$$
F_{(N)}(\theta) = N F(\theta)
$$

Now, due to Cramér–Rao inequality, a locally unbiased estimator
$T(x_1, \ldots, x_N)$
for the joint distribution
$p(x_1)\cdots p(x_N)$
satisfies:

$$
\operatorname{Cov}_\theta(T) \succeq \frac{F(\theta)^{-1}}{N}
$$

This provides a bound on the convergence of a locally unbiased estimator when $N \to \infty$.

## Fisher Metric

Let us introduce the inner product between parameter changes
$d\theta_1, d\theta_2 \in \mathbb{R}^n$
induced by the Fisher matrix:

$$
\langle d\theta_1, d\theta_2 \rangle_F := d\theta_2^T F d\theta_1
$$

Let us call $\langle \cdot , \cdot \rangle_F$ the Fisher metric. Obviously, it is not a metric in the usual elementary sense. However, it has a connection to differential geometry.

One of the main properties of the Fisher metric is its invariance with respect to reparameterization. For theoretical rigor, we only consider diffeomorphisms. By diffeomorphism we mean a $C^1$ map between $\mathbb{R}^n$ and $\mathbb{R}^n$ which is bijective and whose inverse is also $C^1$.

<div class="framed" markdown="1">

<div class="statementinner" markdown="1">

**Statement.**

Fisher metric is invariant under any diffeomorphism $\theta(\eta)$ with Jacobian

$$
J(\eta) = \frac{\partial \theta}{\partial \eta}.
$$

</div>

</div>

#### Proof

Let

$$
J(\eta) := \frac{\partial}{\partial \eta}\theta(\eta)
$$

be the Jacobian of the reparameterization. By the chain rule:

$$
\frac{\partial}{\partial \eta}\log p_{\theta(\eta)}(x)
=
J(\eta)^T \frac{\partial}{\partial \theta } \log p_\theta (x),
\qquad
s_\eta (x) = J(\eta)^T s_{\theta (\eta)}(x)
$$

Fisher matrix in the new coordinates $\eta$ can be written as:

$$
F_\eta (\eta)
=
\mathbb{E}\left[ s_\eta s_\eta^T \right]
=
\mathbb{E} \left[ J(\eta)^T s_{\theta(\eta)}s_{\theta(\eta)}^T J(\eta)  \right]
=
J(\eta)^T F_\theta (\theta(\eta)) J(\eta)
$$

Now let $d\eta_1, d\eta_2$ be infinitesimal parameter changes in $\eta$ space. Then:

$$
d\theta_1 = J(\eta)d\eta_1,
\qquad
d\theta_2 = J(\eta)d\eta_2
$$

Hence:

$$
\langle d\eta_1,d\eta_2\rangle_{F_\eta}
=
d\eta_1^T F_\eta(\eta)d\eta_2
=
d\eta_1^T J(\eta)^T F_\theta(\theta(\eta)) J(\eta)d\eta_2
=
d\theta_1^T F_\theta(\theta(\eta)) d\theta_2
=
\langle d\theta_1,d\theta_2\rangle_{F_\theta}
$$

---

So, Fisher metric does not depend on the particular choice of parameterization, making it an effective inner product between infinitesimal distribution changes. In fact, when $F \succ 0$, Fisher metric is a Riemannian metric on the statistical manifold.

Fisher metric has connections to the KL divergence

$$
KL(p_1 \,\|\, p_2) := \mathbb{E}_1\log \frac{p_1}{p_2}.
$$

Consider a point $\theta \in \mathbb{R}^n$ and a small parameter change $\Delta \theta \in \mathbb{R}^n$. KL divergence between $\theta$ and $\theta + \Delta \theta$ is:

$$
K(\theta, \theta + \Delta \theta)
:=
KL(p_{\theta} \,\|\, p_{\theta + \Delta \theta})
=
\mathbb{E}_\theta \left[ \log \frac{p_\theta}{p_{\theta + \Delta \theta}} \right]
$$

We would like to write the Taylor expansion up to second order. The first term in KL divergence is fixed, and we differentiate with respect to the second term. For that, we compute:

$$
\frac{\partial }{\partial \eta } K(\theta, \eta)
=
\frac{\partial }{\partial \eta } \mathbb{E}_\theta \left[ \log p_\theta - \log p_\eta  \right]
=
-\mathbb{E}_\theta \left[ \frac{\partial }{\partial \eta} \log p_\eta \right]
=
-\mathbb{E}_\theta \left[ s_\eta \right]
$$

$$
\frac{\partial^2}{\partial \eta ^2} K(\theta, \eta)
=
-\mathbb{E}_\theta \left[ \frac{\partial ^2}{\partial \eta^2}\log p_\eta  \right]
$$

The first derivative vanishes at point $\theta$ due to the zero-mean property of the score function. The second derivative becomes $F(\theta)$ by the identity proved earlier.

Thus:

$$
KL(p_\theta \,\|\, p_{\theta + \Delta \theta} )
=
\frac{1}{2}\Delta \theta^T F(\theta) \Delta \theta + O(\|\Delta \theta\|^3)
=
\frac{1}{2}\|\Delta \theta \|^2_F + O(\|\Delta \theta\|^3)
$$

where the norm $\|\cdot \|_F := \sqrt{\langle \cdot, \cdot \rangle_F}$ is induced by the Fisher metric.

So, Fisher metric provides a second-order approximation of the KL divergence. This makes it suitable for trust-region methods with KL constraint.

Next, let us derive an object closely related to the Fisher metric. We can view $\langle \cdot, \cdot \rangle_F$ as an ordinary Euclidean scalar product for transformed vectors:
$v \mapsto F^{1/2}v$.
The volume change induced by this transformation can be expressed with determinant
$\det(F^{1/2}) = \det(F)^{1/2}$.

Jeffreys prior is a prior distribution on parameters $\theta$ that assigns probability mass proportionally to “information volume”:

$$
\pi _J (\theta) \propto \sqrt{\det \left(F(\theta)\right)}
$$

Here, for correct definition, we consider the case where $F(\theta) \succ 0$ for all $\theta$.

<div class="framed" markdown="1">

<div class="statementinner" markdown="1">

**Statement.**

Jeffreys prior is invariant under any diffeomorphism $\theta(\eta)$ with Jacobian

$$
J(\eta) = \frac{\partial \theta}{\partial \eta}.
$$

</div>

</div>

#### Proof

From the proof of Fisher metric invariance:

$$
s_\eta(x,\eta) = J(\eta)^T s_\theta(x,\theta),
\qquad
F_\eta(\eta) = J(\eta)^TF_\theta(\theta(\eta))J(\eta)
$$

Taking determinants, we obtain

$$
\det F_\eta(\eta)
=
\det(J^T F_\theta(\theta) J)
=
\det(J)^2 \det(F_\theta(\theta))
$$

hence

$$
\sqrt{\det F_\eta(\eta)} = |\det J(\eta)| \sqrt{\det F_\theta(\theta)}
$$

For reparameterization of the distribution, we know that:

$$
\pi^*_J(\eta) = |\det J(\eta)|\pi_J(\theta(\eta))
$$

Now it is easy to see that:

$$
\pi_J(\eta)
:=
\sqrt{\det F_\eta(\eta)}
=
|\det J(\eta)| \sqrt{\det F_\theta (\theta)}
=
|\det J(\eta)| \pi_J(\theta)
=
\pi^*_J(\eta)
$$

Thus, Jeffreys prior is invariant under smooth reparameterization.

## Singularity analysis

Usually, statistics books consider the statistical model to always have positive definite Fisher matrix $F \succ 0$, however we would like to provide interpretation of its singularity.

First, let us slightly weaken the definition of the statistical model: $\Theta$ is a set in $\mathbb{R}^n$ (not necessarily open), and the map $\theta \mapsto p_\theta$ at each point $x$ does not have to be bijective, though it is still $C^1(\Theta)$.

Let us call the statistical model **locally overparameterized** at point $\theta \in \Theta$ if there exists a direction $v \in \mathbb{R}^n$ that does not change a distribution in first order:

$$
\left. \frac{d}{dt}p_{\theta + tv}(x) \right|_{t=0} = 0
\quad \forall x \in \mathbb{X}
$$

We can equivalently express this in terms of the score function using directional derivative:

$$
\left. \frac{d}{dt}p_{\theta + tv}(x) \right|_{t=0} = 0
\iff
\left. \frac{d}{dt}\log p_{\theta + tv}(x) \right|_{t=0}
=
\frac{\partial \log p_\theta(x)}{\partial v}
=
\langle s_\theta(x), v \rangle
=
0
$$

<div class="framed" markdown="1">

<div class="statementinner" markdown="1">

**Statement (Criterion of singularity of the Fisher matrix).**

Fisher matrix $F(\theta)$ is singular if and only if the model is locally overparameterized at point $\theta$.

</div>

</div>

#### Proof

$$
\langle Fv, v \rangle
=
\mathbb{E}\left[\langle s_\theta, v \rangle^2\right]
=
0
\iff
\langle s_\theta(x), v \rangle = 0 \quad \forall x
\iff
\text{the model is locally overparameterized at } \theta
$$

---

This simple criterion allows us to intuitively understand that zero eigenvalues of the Fisher matrix correspond to directions that do not change the distribution locally. Hence, they do not contain any “information” about the distribution locally.

When strictly defining a statistical model, local overparameterization does not exist.

<div class="framed" markdown="1">

<div class="statementinner" markdown="1">

**Statement.**

If $\Theta \subset \mathbb{R}^n$ and $\theta \leftrightarrow p_\theta$ is a diffeomorphism, then the model is not locally overparameterized at any point. Equivalently, $F(\theta) \succ 0$ for every $\theta \in \Theta$.

</div>

</div>

#### Proof

Fix any $\theta \in \Theta$. Suppose the model is locally overparameterized at $\theta$. Then there exists a nonzero direction $v \in \mathbb{R}^n$, $v \ne 0$, such that

$$
\left. \frac{d}{dt} p_{\theta + tv}(x) \right|_{t=0} = 0
\qquad \forall x \in \mathbb{X}
$$

Denote by $\Phi(\theta) = p_\theta$ the map from parameters to distributions. Since $\theta \leftrightarrow p_\theta$ is a diffeomorphism, it can never have any directional derivative equal to zero. Therefore,

$$
\left. \frac{d}{dt} p_{\theta + tv} \right|_{t=0}
=
\left. \frac{d}{dt} \Phi(\theta + tv) \right|_{t=0}
\ne 0
$$

which is a contradiction.

<div class="framed" markdown="1">

<div class="statementinner" markdown="1">

**Statement.**

If $T(x)$ is locally unbiased at point $\theta_0$, then the statistical model is not locally overparameterized at point $\theta_0$.

Equivalently, if the model is locally overparameterized at $\theta_0$, a locally unbiased estimator at $\theta_0$ does not exist.

</div>

</div>

#### Proof

Suppose $T(x)$ is a locally unbiased estimator at $\theta_0$ and also the model is locally overparameterized at $\theta_0$: there exists a direction $v \ne 0$ such that for all $x$,

$$
\left. \frac{d}{dt}p_{\theta + tv}(x) \right|_{t=0} = 0
$$

As the estimator is unbiased in some neighborhood of $\theta_0$, for sufficiently small $t$ we have:

$$
\mathbb{E}_\theta \left[T \right] = \theta,
\qquad
\mathbb{E}_{\theta + tv}\left[T\right] = \theta + tv
$$

This implies:

$$
\mathbb{E}_{\theta + tv}\left[T\right] - \mathbb{E}_\theta \left[T \right]
=
\int T(x)\left(p_{\theta + tv}(x) - p_\theta(x)\right)\,dx
=
tv
$$

Dividing by $t$ and taking the limit $t \to 0$:

$$
\int T(x)\left. \frac{d}{dt}p_{\theta + tv}(x) \right|_{t=0}dx
=
0
=
v
$$

This contradiction proves the statement.

---

In neural networks, typically there is redundancy of parameters in a sense that model dimension can be reduced. The final statement of this section naturally explains the behavior of the Fisher matrix for such case.

<div class="framed" markdown="1">

<div class="statementinner" markdown="1">

**Statement.**

Let

$$
\theta \in \Theta = \mathbb{R}^n \mapsto \eta(\theta) \in \Omega \subset \mathbb{R}^m
$$

be a $C^1$ map with Jacobian

$$
J(\theta) := \frac{\partial}{\partial \theta}\eta(\theta) \in \mathbb{R}^{m \times n},
\qquad m < n
$$

where $\Omega$ is an open set. Suppose there is a bijection
$\eta(\theta) \leftrightarrow p_{\eta(\theta)}$
and $\eta(\theta) \mapsto p_{\eta(\theta)}(x)$ is $C^{\infty}$ for all $x \in \mathbb{X}$.
In other words, $p_{\eta(\theta)}$ is an $m$-dimensional statistical model in the strict sense.

Then if we consider the $n$-dimensional parameterization
$\theta \mapsto p_\theta := p_{\eta(\theta)}$,
and define the $n \times n$ Fisher matrix

$$
F(\theta) := \mathbb{E}_\theta \left[s_\theta s_\theta^T \right],
$$

the following holds:

$$
\operatorname{rank}(F(\theta)) = \operatorname{rank}(J(\theta)) \le m
$$

In particular, when Jacobian $J(\theta)$ has full row rank $m$ for every $\theta$, Fisher matrix $F(\theta)$ always has rank $m$.

</div>

</div>

#### Proof

From the chain rule, similarly to the reparameterization formula, we have:

$$
F(\theta) = J(\theta)^T F_{\eta}(\eta(\theta))J(\theta)
$$

Because $\eta(\theta) \leftrightarrow p_{\eta(\theta)}$ is a diffeomorphism, the $m \times m$ Fisher matrix $F_{\eta}(\eta(\theta))$ is positive definite, and therefore has full rank $m$.

Now the following sequence of equivalences shows that
$\ker F(\theta) = \ker J(\theta)$
and therefore their ranks are equal:

$$
F(\theta)v = 0
\iff
\langle F(\theta)v, v\rangle = 0
\iff
\left\langle F_\eta(\eta(\theta)) \left[J(\theta) v\right],\, J(\theta)v\right\rangle = 0
\overset{F_\eta \succ 0}{\iff}
J(\theta) v = 0
$$

Obviously, $\operatorname{rank}(J(\theta)) \le m$ because the matrix is $m \times n$. So the statement follows.

---

This provides a good way to understand Fisher matrix rank. If the statistical model induced by parameters has dimension $m$, then in any higher-dimensional parameterization the rank will be at most $m$.

Let us consider an example which shows how a “bad” parameterization can lead to Fisher matrix with rank less than $m$. Take a point
$\theta = (\theta_1, \theta_2) \in \mathbb{R}^2$
that defines the normal distribution
$\mathcal{N}(\theta_1^3, 1)$:

$$
p_\theta(x) = \frac{1}{\sqrt{2\pi}}\exp\left(-\frac{(x-\theta_1^3)^2}{2}\right)
$$

This distribution can be naturally parameterized with
$\eta(\theta) = \theta_1^3$,
and it will have constant Fisher matrix:

$$
F_\eta(\eta(\theta)) \equiv 1
$$

Now, let us compute the score and original Fisher matrix:

$$
s_\theta(x)
=
\frac{\partial}{\partial \theta} \left[\text{const} - \frac{1}{2}(x - \theta_1^3)^2\right]
=
\begin{bmatrix}
-3(x-\theta_1^3)\theta_1^2 \\
0
\end{bmatrix}
$$

$$
F_\theta(\theta)
=
\begin{bmatrix}
9\theta_1^4\mathbb{E}_\theta \left[ (x - \theta_1^3)^2 \right] & 0 \\
0 & 0
\end{bmatrix}
=
\begin{bmatrix}
9\theta_1^4 & 0 \\
0 & 0
\end{bmatrix}
=
\begin{bmatrix}
3\theta_1^2 \\
0
\end{bmatrix}
\cdot 1 \cdot
\begin{bmatrix}
3\theta_1^2 & 0
\end{bmatrix}
$$

Now, we can see that at point $\theta_1 = 0$ Fisher matrix becomes zero with rank $0$, and the Jacobian

$$
J(\theta)
=
\begin{bmatrix}
3\theta_1^2 \\ 0
\end{bmatrix}
$$

also has rank $0$ at that point.

It is easy to show that if we perform a one-to-one (non-diffeomorphic) mapping
$\theta_1 \leftrightarrow \theta_1^3$,
then the new model will always have rank-$1$ Jacobian and therefore Fisher will also always have rank $1$.

This example shows that for some strangely parameterized models the Fisher matrix can have rank even less than the dimension of the statistical model. But usually they are the same, and rank of Fisher matrix equals the dimension of the statistical model.

## References

- S. Amari and H. Nagaoka, *Methods of Information Geometry*, 2000.
- H. Jeffreys, “An invariant form for the prior probability in estimation problems,” *Proceedings of the Royal Society A*, 1946.
