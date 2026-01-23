# Zero Redundancy Training

**Setting**
* One Model
  * One set of weights
* Large dataset
* Multiple GPUs (thousands of GPUs)


This technique builds on the discussion of Data Parallelism (DP) and Model Parallelism (MP).

Here, we will trade off the synchronization of data between GPUs for actual memory use.

Recall: Memory use
* Model weights (2-4 N bytes) - *2 if you use bf16, 4 if you use regular fp32*
* Gradients (2-4 N bytes)
* Optimizer State (6-8 N bytes) - *these are the momentum terms*

Optimizer state only used once per step. 
*i.e. only after completing the entire forward and backward pass. We only need the first and second momentu terms of the optimizer, after we synchronize the gradients.*
* Why not distribute this across GPUs?

---

## Zero-1

**Optimizer State Partitioning**

* For M GPUs, each GPU
  * bfloat16 weights (2 N bytes)
    - *store a copy on each GPU*
  * bfloat16 gradients (2 N bytes)
    - *store a copy on each GPU*
  * bfloat32 weights, first, second momentum (3 * 4 N / M bytes)
    - *split the first & second momentum and store **only one part** on each GPU*
    - *Zero-1 stores **two copies** of the weights, which is not necessarily required*
* Forward / backward in bfloat16
* Optimizer step: gather gradients, scatter updated weights

* Reduces memory consumption
  * 12-16 N bytes -> 4 N + 12 N / M bytes - *12 N has to do with Zero-1 stores **two copies** of the weights*
  * 3-4 X for large enough M - *this allows us to train models that are **3 to 4 times** larger than we could otherwise*
* Requires synchronization through **reduce-scatter** algorithm (NCCL library).
  - It sums up gradients across multiple GPUs, and automatically split and distribute the weights onto different GPUs.

* Forward / backward in bfloat16
* Optimizer step: gather gradients, scatter updated weights

---

## Zero-2

**Optimizer State Partitioning**

* Distributed gradient among GPUs
* In backward
  - Compute gradient
  - **all-reduce**
  - Keep gradient corresponding to GPUs optimizer state

* Reduces memory consumption
  * 12-16 N bytes -> 2 N + 16 N / M bytes - *its 16 N because it stores weights and gradients in higher precision*
  * 6-8 X large enough M - *this allows us to train models that are **6 to 8 times** larger than we could otherwise*
* Requires synchronization through **reduce-scatter**, **all-reduce**
  * *we need to call all-reduce every time we calculate a gradient, because we no longer have the memory to store all gradients on the current GPU*

---

## Zero-3

**Optimizer State Partitioning**

* Only store weights that are currently required
* In forward / backward
  * **all-gather**
  * Compute
  * Discard weights
* Use bf16 weights or original fp32

---

# Reduce-Scatter Algorithm

