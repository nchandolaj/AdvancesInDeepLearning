# Data Parallelism (DP)

In 2026, **Data Parallelism (DP)** remains the most widely used distributed training strategy. It is the "bread and butter" of AI scaling because it is conceptually simple and provides a nearly linear speedup for models that can fit on a single GPU.

### 1. The Core Concept: "Divide the Data, Copy the Model"
In Data Parallelism, the **model is replicated** (copied) across multiple GPUs, but the **dataset is sharded** (split). Each GPU receives a different "mini-batch" of data and processes it independently.

#### The Lifecycle of a DP Training Step:
1.  **Replication:** Every GPU starts with the exact same model weights.
2.  **Scattering:** The main training batch is split into $N$ pieces (where $N$ is the number of GPUs).
3.  **Forward Pass:** Each GPU calculates its own local "loss" based on its unique slice of data.
4.  **Backward Pass:** Each GPU calculates its own local "gradients" (instructions on how to change the weights to reduce the loss).
5.  **Synchronization (The All-Reduce):** This is the critical step. The GPUs communicate to average their gradients. (e.g., if GPU 1 thinks a weight should increase by 0.2 and GPU 2 thinks it should decrease by 0.1, they agree on a global average of +0.05).
6.  **Update:** Every GPU updates its local weights using this unified average, ensuring all replicas remain identical for the next round.

### 2. Advantages of Data Parallelism
* **Near-Linear Speedup:** If you have 8 GPUs, your training can theoretically happen nearly 8x faster than on a single GPU (minus the time spent "talking" to each other).
* **Ease of Implementation:** Modern frameworks like PyTorch (`DDP`) and TensorFlow (`MirroredStrategy`) allow you to turn a single-GPU script into a multi-GPU script with just 2–3 lines of code.
* **Statistical Stability:** By combining gradients from many GPUs, you are effectively training with a much larger "Global Batch Size," which can lead to smoother and more stable convergence.

### 3. Shortcomings & Challenges
* **Memory Redundant:** Because every GPU holds a full copy of the model, DP is inefficient for massive models. If a model takes 100GB of VRAM and your GPUs only have 80GB, **Standard DP will fail** because the model cannot fit on a single chip.
* **The "Straggler" Problem:** In synchronous DP, the entire cluster must wait for the slowest GPU to finish its math. If one network cable is slightly loose or one GPU is overheating, the whole 1,000-chip cluster slows down.
* **Communication Bottleneck:** As you add more GPUs, the time spent averaging gradients (the All-Reduce) grows. Eventually, you spend more time talking than "thinking."

### 4. Modern Evolution: DP vs. DDP vs. FSDP
In the early days, "DataParallel" (DP) was the standard, but it was slow because it used a single "master" process to collect all the data. By 2026, two evolved versions have taken over:

| Technique | How it works | When to use it |
| :--- | :--- | :--- |
| **Distributed Data Parallel (DDP)** | Spawns a separate process for every GPU. This avoids "bottlenecks" at the master CPU. | **Standard choice** for models that fit on one GPU. |
| **Fully Sharded DP (FSDP)** | Breaks the model into pieces and only loads the piece needed for a specific layer's math. | **Required** for models too large for one GPU (e.g., 70B+ parameter models). |


### Summary Table: Data Parallelism vs. Model Parallelism

| Feature | Data Parallelism | Model Parallelism |
| :--- | :--- | :--- |
| **What is split?** | The Data | The Model Layers/Tensors |
| **Model Copy** | Full copy on every GPU | Sliced across GPUs |
| **Primary Goal** | Increase **Speed** | Increase **Model Size** |
| **Communication** | Gradients (End of step) | Activations (During step) |

