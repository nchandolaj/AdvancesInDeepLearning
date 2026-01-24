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

**Synchronization:** One synchronization, after each forward and backwards call, to synchronize our weights and gradients. 
* Requires synchronization through **reduce-scatter** algorithm (NCCL library).
* It sums up gradients across multiple GPUs, and automatically split and distribute the weights onto different GPUs.

**Execution:**
* Forward / backward in bfloat16
* Optimizer step: gather gradients, scatter updated weights

---

## ZeRO-2 Or Stage 2: Gradient Partitioning ($P_g$)

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
* 12-16 N bytes -> 2 N + 16 N / M bytes - *its **16 N** because in ZeRO-1/2/3, the **weights and gradients are stored in higher precision** on each GPU*
* 6-8 X large enough M - *this allows us to train models that are **6 to 8 times** larger than we could otherwise*

**Synchronization:** Synchronization called every single time we compute the gradient, because we dont have the memory to store all gradients on a single GPU.
* Requires synchronization through **reduce-scatter**, **all-reduce**
* Need to distribute all the updated weights to all the GPUs
* Call all-reduce every time we calculate a gradient, because we no longer have the memory to store all gradients on the current GPU, and synchronize the gradient.

---

## ZeRO-3 Or Stage 3: Parameter Partitioning ($P_p$)

### How it works
* This is the most aggressive stage. The **model parameters are partitioned**.
* When a layer needs to be computed during the forward or backward pass, the **GPU "gathers" the missing parameters from other GPUs**, performs the calculation, and then immediately discards the non-local parameters.

### Benefit
Memory consumption **scales linearly** with the number of GPUs. This allows you to train models that are far larger than the memory of any single GPU.

### Detailed Discussion

Each GPU only keeps a subset of the weights, a subset of the gradients, and a subset of the momentums.
Every single time we call forward or backward, every time we need the weights, we synchronize them between all the GPUs. 

* Only store weights that are currently required
* In forward / backward
  * **all-gather**
  * Compute
  * Discard weights
* Use bf16 weights or original fp32

**Memory requirements:** 
* 12-16 N bytes -> 16 N / M bytes - *its **16 N** because in ZeRO-1/2/3, the **weights and gradients are stored in higher precision** on each GPU*
* 50+ X for large enough M - *this allows us to train models that are more than **50 X times** in the size of the model than we could otherwise*

**Synchronization:** Synchronization called every time we need the weights when we call forward or we call backwards.
* Requires synchronization through **reduce-scatter**, **all-reduce**, **all-gather** for each layer.
* With this, the main bottleneck is jsut sending chunks of our model from GPU to GPU. Previously, memory was the main bottleneck.

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

* **ZeRO-1** and **ZeRO-2** are quite popular. 
* **ZeRO-3** implementation is sub-optimal.
* People use **Fully Sharded Data Parallelism** more often, which use the same idea as **ZeRO-3** but is **less synchronization** heavy.

---

### ZeRO FSDP: Fully Sharded Data Parallel

Used to train the largest llama models at Meta, and other companies for large models.

**FSDP**: Efficient implementation of ZeRO-3 in PyTorch
  - Synchronize groups (Units) of layers.
  - Efficient scheduling of communication and computation.

**Synchronization**
- Synchronization of gradients happens in groups of forward and backward call, Not after each individual forward and backward call,
- More communication efficient.
- Some scheduling improvements as well.

**Hybrid Sharding**
* 8-16 GPUs per server
* Shard within server. We split the model across a smaller subset of GPUs (on the same node or nodes closer to each other), instead of across all the GPUs we have.
* Regular data parallel (share gradient) between servers (different groups)

**Memory requirements**
* ZeRO / FSDP
* 16 N / M bytes without counting activations
* For M GPUs
  * Good solution for GPU-rich people

---

## Advanced ZeRO Extensions

Beyond the three standard stages, the ecosystem has expanded to handle even more extreme constraints:

* **ZeRO-Offload:** Moves optimizer states and gradients from the GPU to the **CPU RAM**. This allows a single GPU to train models that would normally require a cluster.

* **ZeRO-Infinity:** Extends offloading even further to **NVMe storage**, enabling the training of "trillion-parameter" models by utilizing all available memory across the entire system (GPU, CPU, and Disk).

* **ZeRO++:** Adds optimizations like **quantization** and **hierarchical partitioning** to reduce the communication overhead introduced by Stage 3, making it faster on slower networks.

---

# ZeRO-1 Reduce-Scatter Algorithm - Workflow Steps

In **ZeRO-1**, the goal is to eliminate the redundancy of **Optimizer States** (which, for Adam, are the largest memory consumers). The **Reduce-Scatter** algorithm is the engine that makes this possible. 

The **Reduce-Scatter** algorithm ensures that while every GPU participates in the calculation of gradients, only one "owner" GPU keeps the final averaged result for a specific shard of the model.

Here are the workflow steps of the Reduce-Scatter algorithm during a ZeRO-1 training iteration, assuming a setup with **$N$ GPUs**.

## 1. Local Gradient Generation (The Backward Pass)
Before any communication happens, every GPU works independently on its own micro-batch of data.
* **Within the GPU:** The GPU performs the standard backward pass, calculating the gradients for every single parameter in the model.
* **Data State:** Each GPU now holds a full set of "local gradients" (let's call them $G_{local}$). These gradients are unique to the specific data samples that the GPU just processed.
* **Communication:** None yet.

## 2. Gradient Sharding (The Logical Split)
To prepare for Reduce-Scatter, the total model gradients are logically divided into $N$ equal-sized blocks or "shards" (e.g., $B_0, B_1, \dots, B_{n-1}$).
* **Within the GPU:** The GPU identifies which shard it is responsible for "owning." For example, GPU 0 is the owner of $B_0$.
* **Communication:** Preparation for the transfer.

## 3. The Reduction Phase (Summing Across Nodes)
This is the "Reduce" part of the name. The GPUs must combine their local gradients so the result is an average of the entire training batch.
* **What it Sends:** Each GPU sends its local gradient shards to the respective owners. For instance, GPU 1 sends its $B_0$ shard to GPU 0, its $B_2$ shard to GPU 2, and so on.
* **What it Receives:** Simultaneously, each GPU receives the corresponding shards from all other GPUs. GPU 0 receives the $B_0$ shards from GPUs 1, 2, and 3.
* **Within the GPU:** As the shards arrive, the GPU performs an element-wise summation (or averaging).
    * *Example:* GPU 0 sums the $B_0$ from every other GPU: $\sum B_0 = B_0(G_0) + B_0(G_1) + B_0(G_2) + B_0(G_3)$.

## 4. The Scatter Phase (Discarding Redundancy)
This is the "Scatter" part of the name. Once the summation is complete, the "global" gradient for that specific shard is localized.
* **Within the GPU:** Once GPU 0 has the fully reduced (averaged) $B_0$, it **discards** all other gradient shards ($B_1, B_2, B_3$) from its memory. 
* **Data State:** GPU 0 now only holds the averaged gradients for the first $1/N$ of the model. 
* **Communication:** This concludes the "Scatter" because the result of the reduction is now scattered across the cluster—each GPU holding exactly one unique piece.

## 5. Optimizer Update (The ZeRO-1 Specific Step)
Now that the gradients are reduced and scattered, the optimizer states (which are already partitioned in ZeRO-1) can be applied.
* **Within the GPU:** GPU 0 looks at its local **Optimizer States** (which only exist for $B_0$) and applies them to its averaged gradients ($B_0$). It updates its local copy of the **Parameters** for that shard.
* **Communication:** None. The update is purely local to the shard.

### Summary of Data Flow for GPU $i$
| Step | Action | Sent to Others | Received from Others |
| :--- | :--- | :--- | :--- |
| **Backward** | Calculate full gradients | Nothing | Nothing |
| **Reduce** | Sum specific shard $B_i$ | All shards $B_{\neq i}$ | Shards $B_i$ from all $N-1$ GPUs |
| **Scatter** | Delete $B_{\neq i}$ | Nothing | Nothing |
| **Update** | Update local param shard | Nothing | Nothing |

---

# ZeRO-2 Algorithm: All-Reduce - Workflow Steps (for Zero-2)

In **ZeRO-2**, the standard **All-Reduce** is not treated as a single monolithic operation. Instead, it is executed as a composite of two distinct phases: **Reduce-Scatter** and **All-Gather**. 

This "deconstructed All-Reduce" is the secret to ZeRO-2's efficiency. It allows the system to synchronize gradients across the cluster while ensuring that no single GPU ever has to store the full set of gradients or optimizer states simultaneously.

## 1. Phase One: Reduce-Scatter (Gradient Sync)
The goal of this phase is to sum the gradients across all GPUs but "scatter" the result so each GPU only keeps the portion it is responsible for.

### Step 1: Local Backward Pass
* **Within the GPU:** The GPU completes the backward pass on its local micro-batch. It now holds the local gradients ($G_{local}$) for the **entire model**.
* **Communication:** None.

### Step 2: The Scatter-Reduce Exchange
* **What it Sends:** Each GPU divides its gradients into $N$ shards. It sends shard $B_1$ to GPU 1, shard $B_2$ to GPU 2, etc.
* **What it Receives:** Simultaneously, it receives its own assigned shard (e.g., GPU 0 receives all $B_0$ shards) from every other GPU in the cluster.
* **Within the GPU:** As shards arrive, the GPU performs an **in-place summation**.

### Step 3: Gradient Deletion
* **Within the GPU:** Once the summation for its own shard is complete, the GPU **deletes** the gradients for all other shards. 
* **Result:** GPU 0 now only holds the averaged gradients for Shard $B_0$. Memory for $B_1 \dots B_n$ is now free.

## 2. Phase Two: Local Optimizer Update
Before the "All-Reduce" can be completed (via All-Gather), the GPU must update the actual weights.

* **Within the GPU:** GPU 0 takes its averaged gradient shard ($B_0$) and its local optimizer states (momentum/variance for $B_0$) and performs the update (e.g., Adam update).
* **Result:** The local parameter shard $W_0$ is now updated. However, the GPU still has "old" weights for $W_1 \dots W_n$.
* **Communication:** None.

## 3. Phase Three: All-Gather (Parameter Sync)
To complete the All-Reduce cycle, the GPUs must share their updated parameter shards so everyone has the full, updated model for the next forward pass.

### Step 1: The Ring Exchange
* **What it Sends:** GPU 0 sends its updated $W_0$ to GPU 1.
* **What it Receives:** GPU 0 receives $W_{n}$ (the last shard) from the last GPU in the ring.
* **Within the GPU:** The GPU places the received shard into the correct "slot" in its parameter buffer.

### Step 2: Propagation
* **Communication:** This continues in a "ring" for $N-1$ steps until every shard has visited every GPU.
* **Result:** Every GPU now has a complete, identical copy of the updated parameters ($W_{total}$).

## Summary of Data Movement in ZeRO-2 "All-Reduce"

| Step | GPU $i$ Sends | GPU $i$ Receives | End State of GPU $i$ |
| :--- | :--- | :--- | :--- |
| **Reduce-Scatter** | Shards $B_{j \neq i}$ | Shards $B_i$ from all peers | Has averaged gradients for $B_i$ only. |
| **Update** | Nothing | Nothing | Has updated parameters for $W_i$ only. |
| **All-Gather** | Shard $W_i$ (and subsequent) | Shards $W_{j \neq i}$ | Has full updated model parameters. |

### Why this is better than "Standard" All-Reduce
In a standard All-Reduce, the GPU would hold **Full Gradients + Full Optimizer States + Full Parameters**. 
In ZeRO-2, the GPU only holds **$1/N$ Gradients + $1/N$ Optimizer States + Full Parameters**. Since optimizer states (especially in Adam) are often 3x-4x larger than the parameters themselves, this represents a massive reduction in the memory floor.

---

# ZeRO-3 All-Gather Algorithm: Workflow Steps

In **ZeRO-3**, the **All-Gather** algorithm is fundamentally different from ZeRO-2. In ZeRO-2, the All-Gather happens once per iteration to reconstruct the whole model. In ZeRO-3, the model parameters are **permanently partitioned**. 

To save maximum memory, ZeRO-3 performs "Just-in-Time" (JIT) All-Gathers. It fetches only the parameters needed for a specific layer, performs the math, and then **deletes them immediately** before moving to the next layer.

## The Workflow: Layer-by-Layer All-Gather

### 1. The Trigger (Forward or Backward Pass)
Training begins, and the engine reaches a specific layer (e.g., Layer 5).
* **Within the GPU:** Every GPU realizes it only has $1/N$ of the parameters for Layer 5. It cannot perform the computation (matrix multiplication) yet.
* **Communication:** A collective request is issued to start an All-Gather specifically for the parameters of Layer 5.

### 2. The Step-by-Step Exchange (The Ring)
To reconstruct the layer, the GPUs use a ring-based All-Gather:

* **Step 1: The Initial Push:** Each GPU sends its local shard of Layer 5's parameters to its neighbor in the ring.
* **Step 2: Propagation:** For $N-1$ steps, each GPU receives a shard it was missing and passes it to the next neighbor. 
* **Within the GPU:** As the shards arrive, the GPU stitches them together in a temporary "buffer" or "scratchpad" memory.

### 3. Computation and Immediate Eviction
* **Within the GPU:** Once the temporary buffer contains the **full parameters for Layer 5**, the GPU performs the forward pass (calculating activations) or backward pass (calculating gradients).
* **The "ZeRO-3 Twist" (Eviction):** As soon as the math for Layer 5 is done, the GPU **deletes the gathered parameters**, keeping only its original $1/N$ shard.
* **Communication:** None.

### 4. Moving to the Next Layer
The process repeats for Layer 6. The GPUs All-Gather Layer 6, use them, and delete them.
* **Memory State:** The GPU's memory usage remains low and flat because it only ever holds the full parameters of **one layer at a time**, rather than the whole model.

## Comparison of All-Gather: ZeRO-2 vs. ZeRO-3

| Feature | ZeRO-2 All-Gather | ZeRO-3 All-Gather |
| :--- | :--- | :--- |
| **Frequency** | Once per training step (after update). | **Twice per layer** (once in forward, once in backward). |
| **Data Scope** | Entire model parameters. | Single layer/sub-module parameters. |
| **Persistence** | Parameters stay on GPU for the whole step. | Parameters are **deleted** immediately after use. |
| **Communication** | Lower overhead, higher memory. | Higher overhead, lowest possible memory. |

## Summary of Data Flow for GPU $i$ (Layer $L$)

| Phase | Sent to Neighbors | Received from Neighbors | Internal Action |
| :--- | :--- | :--- | :--- |
| **Pre-Compute** | Shard $W_{L,i}$ | Missing shards $W_{L,j}$ | Reconstruct Layer $L$ in temp buffer. |
| **Compute** | Nothing | Nothing | Run Layer $L$ math. |
| **Post-Compute** | Nothing | Nothing | **Wipe temp buffer.** Only $W_{L,i}$ remains. |

