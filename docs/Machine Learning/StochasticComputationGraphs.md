# Gradient Estimation using Stochastic Computation Graphs

### Abstract

In a variety of problems originating in supervised, unsupervised, and reinforcement learning, the loss function is defined by an expectation over a collection of random variables, which might be part of a probabilisitc model or the external world. Estimating the gradient of this loss function ,using samples, lies at the core of gradient-based learning algorithms for these problems. We introduce the formalism of stochastic computation graphs - directed acyclic graphs that include both deterministic functions and conditional probability distributions - and describe how to easily and automatically derive an unbiased estimator of the loss function's gradient. The resulting algorithm for computing the gradient estimator is a simple modification of the standard backpropagation algorithm. The generic scheme proposed here unifies estimators derived in variety of prior work, along with variance-reduction techniques therein. It could assist researchers in developing intricate models involving a combination of stochastic and deterministic operations, enabling, for example, attention, memory and control actions.


### Introduction

The great success of neural networks is due in part to the simplicity of the backpropagation algorithm, which allows one to efficiently compute the gradient of any loss function defined as a composition of differentiable functions. This simplicity has allowed researchers to search in the space of architectures for those that are both highly expressive and conducive to optimization; yielding, for example, convolutional neural networks in vision and LSTMs for sequence data. However, the backpropagation algorithm is only sufficient when the loss function is a deterministic, differentiable function of the parameter vector.

A rich class of problems arising throughout machine learning requires optimizing loss functions that involve an epectation over random variables. Two broad categories of these problems are (1) likelihood maximization in probabilistic models with latent variables, and (2) policy gradients in reinforcement learning. Combining ideas from those two perennial topics, recent models of attention and memory have used networks that involve a combination of stochastic and deterministic operations.

In most of these problems, from probabilistic modeling to reinforcement learning, the loss functions and their gradients are intractable,as they involve either a sum over an exponential number of latent variable configurations, or high-dimensional integrals that have no analytic solution. Prior work has provided problem-specific derivations of Monte-Carlo gradient estimators, however, no previous work addresses the general case.

Several classic and recent techniques in variational inference and reinforcement learning, where the loss functions can be straightforwardly described using the formalism of stochastic computation graphs that we introduce. 
The contributions of this work are as follows:
• We introduce a formalism of stochastic computation graphs, and in this general setting, we derive
unbiased estimators for the gradient of the expected loss.
• We show how this estimator can be computed as the gradient of a certain differentiable function
(which we call the surrogate loss), hence, it can be computed efficiently using the backpropagation algorithm. This observation enables a practitioner to write an efficient implementation using automatic differentiation software.
• We describe variance reduction techniques that can be applied to the setting of stochastic computation graphs, generalizing prior work from reinforcement learning and variational inference.
• We briefly describe how to generalize some other optimization techniques to this setting: majorization-minimization algorithms, by constructing an expression that bounds the loss function; and quasi-Newton / Hessian-free methods, by computing estimates of Hessian-vector products.

The main practical result of this article is that to compute the gradient estimator, one just needs to make a simple modification to the backpropagation algorithm, where extra gradient signals are introduced at the stochastic nodes. Equivalently, the resulting algorithm is just the backpropagation algorithm, applied to the surrogate loss function, which has extra terms introduced at the stochastic nodes. 

### 2. Peliminaries

##### 2.1 Gradient Estimators for a Single Random Variable

The SF and PD estimators are applicable in different scenarios and have different properties.
1. SF is valid under more permissive mathematical conditions than PD. SF can be used if f is discontinuous, or if x is a discrete random variable.

2. SF only requires sample values f (x), whereas PD requires the derivatives f′(x). In the context of control (reinforcement learning), SF can be used to obtain unbiased policy gradient estimators in the “model-free” setting where we have no model of the dynamics, we only have access to sample trajectories.

3. SF tends to have higher variance than PD, when both estimators are applicable . The variance of SF increases (often linearly) with the dimensionality of the sampled variables. Hence, PD is usually preferable when x is high-dimensional. On the other hand, PD has high variance if the function f is rough, which occurs in many time-series problems due to an “exploding gradient problem” / “butterfly effect”.

4. PD allows for a deterministic limit, SF does not. This idea is exploited by the deterministic policy gradient algorithm.

**Nomenclature** The methods of estimating gradients of expectations have been independently proposed in several different fields, which use differing terminology. What we call the *score function*
estimator is alternatively called the *likelihood ratio estimator* and **REINFORCE**. We chose this term because the score function is a well-known object in statistics. What we call the **pathwise derivative estimator** (from the mathematical finance literature and reinforcement learning) is alternatively called *infinitesimal perturbation analysis* and *stochastic backpropagation*. We chose this term because pathwise derivative is evocative of propagating a derivative
through a sample path.


##### 2.2 Stochastic Computation Graphs

The results of this article will apply to stochastic computation graphs, which are defined as follows:

Definition 1 (Stochastic Computation Graph). A directed, acyclic graph, with three types of nodes:
1. Input nodes, which are set externally, including the parameters we differentiate with respect to.
2. Deterministic nodes, which are functions of their parents.
3. Stochastic nodes, which are distributed conditionally on their parents.

Each parent v of a non-input node w is connected to it by a directed edge (v, w).