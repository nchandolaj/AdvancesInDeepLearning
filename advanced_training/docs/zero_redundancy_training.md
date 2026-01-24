# Zero Redundancy Training

Training large deep learning models (like LLMs) often hits a "memory wall." 

In traditional **Data Parallelism (DP)**, every GPU keeps a complete copy of the **model parameters, gradients, and optimizer states** (momenturm and variance buffers). This redundancy becomes impossible to manage as models grow to billions of parameters.

**Setting**
* One Model
  * One set of weights
* Large dataset
* Multiple GPUs (thousands of GPUs)


## ZeRO (Zero Redundancy Optimizer)

**Zero Redundancy Training technique** builds on the discussion of **Data Parallelism (DP)** and **Model Parallelism (MP)**.

**ZeRO** tradeoffs the synchronization of data between GPUs for actual memory use.

**Memory requirements (using mixed-precision, DP and/or MP)**
* Model weights (2-4 N bytes) - *2 if you use bf16, 4 if you use regular fp32*
* Gradients (2-4 N bytes)
* Optimizer State (6-8 N bytes) - *these are the momentum terms*

**ZeRO**, developed by Microsoft as part of the **DeepSpeed** library, solves this by **partitioning these states across the available GPUs instead of replicating them**.  

## The Three Stages of ZeRO

ZeRO is typically implemented in **three progressive stages**, each reducing memory redundancy further.
* **ZeRO-1** or **Stage-1**: Optimizer State Partitioning ($P_{os}$)
* **ZeRO-2** or **Stage-2**: Gradient Partitioning ($P_g$)
* **ZeRO-3** or **Stage-3**: Parameter Partitioning ($P_p$)

---

## ZeRO-1 Or Stage 1: Optimizer State Partitioning ($P_{os}$)

In optimizers like Adam, the "optimizer states" (momentum and variance buffers) often take up much more memory than the weights themselves. 

Optimizer state is only used once per step. 
*i.e. only after completing the entire forward and backward pass. We only need the first and second momentum terms of the optimizer, after we synchronize the gradients.*
* Why not distribute this across GPUs?

### How it works
* Each GPU still stores all parameters and gradients, but the **optimizer states are split across GPUs**. 

### Benefits
* Reduces memory usage by up to **4x** with no additional communication overhead compared to standard DP.

### Detailed Discussion

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

**Memory requirements:** Reduces memory consumption
  * 12-16 N bytes -> 4 N + 12 N / M bytes - *12 N has to do with Zero-1 stores **two copies** of the weights*
  * 3-4 X for large enough M - *this allows us to train models that are **3 to 4 times** larger than we could otherwise*

**Synchronization:** Requires synchronization through **reduce-scatter** algorithm (NCCL library).
* It sums up gradients across multiple GPUs, and automatically split and distribute the weights onto different GPUs.

**Execution:**
* Forward / backward in bfloat16
* Optimizer step: gather gradients, scatter updated weights

---

## Stage 2: Gradient Partitioning ($P_g$)

### How it works
* Building on Stage 1, the **gradients are also partitioned**.
* This is trickier than ZeRO-1 because
  - Each GPU still does a full forward and full backward. For this each GPU, only weights are required.
  - However, they also compute gradients and we need to handle those.
  - In the backward pass, we split the gradient across multiple GPUs
* After the backward pass, **each GPU only keeps the gradients corresponding to its specific portion** of the optimizer states.

### Benefits
* Combined with Stage 1, this reduces memory usage by up to **8x**

### Detailed Discussion

* Distributed gradient among GPUs
* In backward
  - Compute gradient
  - **all-reduce** algorithm
    - take the gradient computed for a specific layer on all the GPUs, sum it up and store on each GPU.
    - keep that part of the gradient we are responsible for  in each GPU. Throw all the rest.
  - Keep gradient corresponding to GPUs optimizer state

**Memory requirements:** Reduces memory consumption
* 12-16 N bytes -> 2 N + 16 N / M bytes - *its 16 N because it stores weights and gradients in higher precision*
* 6-8 X large enough M - *this allows us to train models that are **6 to 8 times** larger than we could otherwise*

**Synchronization:** Requires synchronization through **reduce-scatter**, **all-reduce**
* Call all-reduce every time we calculate a gradient, because we no longer have the memory to store all gradients on the current GPU.

---

## Stage 3: Parameter Partitioning ($P_p$)

### How it works
* This is the most aggressive stage. The **model parameters are partitioned**.
* When a layer needs to be computed during the forward or backward pass, the **GPU "gathers" the missing parameters from other GPUs**, performs the calculation, and then immediately discards the non-local parameters.

### Benefit
Memory consumption **scales linearly** with the number of GPUs. This allows you to train models that are far larger than the memory of any single GPU.

### Detailed Discussion

* Only store weights that are currently required
* In forward / backward
  * **all-gather**
  * Compute
  * Discard weights
* Use bf16 weights or original fp32

**Memory requirements:** 


**Synchronization:**


---

## Comparison: Memory Usage

To understand why ZeRO is necessary, consider a model with $\Psi$ parameters using Mixed Precision (FP16/32) training:

| State | Memory Needed (Traditional DP) | Memory Needed (ZeRO-3) |
| :--- | :--- | :--- |
| **Parameters** | $2\Psi$ (FP16) | $2\Psi / N$ |
| **Gradients** | $2\Psi$ (FP16) | $2\Psi / N$ |
| **Optimizer States** | $12\Psi$ (FP32 Adam) | $12\Psi / N$ |
| **Total per GPU** | **$16\Psi$** | **$16\Psi / N$** |

*(Where $N$ is the number of GPUs/devices)*

---

## Advanced ZeRO Extensions

Beyond the three standard stages, the ecosystem has expanded to handle even more extreme constraints:

* **ZeRO-Offload:** Moves optimizer states and gradients from the GPU to the **CPU RAM**. This allows a single GPU to train models that would normally require a cluster.

* **ZeRO-Infinity:** Extends offloading even further to **NVMe storage**, enabling the training of "trillion-parameter" models by utilizing all available memory across the entire system (GPU, CPU, and Disk).

* **ZeRO++:** Adds optimizations like **quantization** and **hierarchical partitioning** to reduce the communication overhead introduced by Stage 3, making it faster on slower networks.

---

# Algorithm: Reduce-Scatter



---

# Algorithm: Reduce-Scatter


---

# Algorithm: Reduce-Scatter


