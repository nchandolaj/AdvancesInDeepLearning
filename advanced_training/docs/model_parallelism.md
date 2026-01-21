# Model Parallelism

When models grow too large to fit into the memory of a single GPU, we must move beyond Data Parallelism and into **Model Parallelism (MP)**. In this strategy, we don't copy the model; we slice it into pieces and distribute those pieces across different chips.

In 2026, Model Parallelism is typically split into two distinct engineering techniques: **Pipeline Parallelism** and **Tensor Parallelism**.


## 1. Pipeline Parallelism (Layer-wise Splitting)
Think of this like an assembly line in a factory. If a model has 80 layers, we might put Layers 1–40 on **GPU A** and Layers 41–80 on **GPU B**.

* **The Workflow:** Data enters GPU A, goes through the first 40 layers, and the resulting "activations" (intermediate results) are sent over the network to GPU B to finish the job.
* **The Challenge (The "Bubble"):** While GPU B is working on the second half, GPU A is sitting idle waiting for the next batch. 
* **The 2026 Solution (Micro-batching):** We break the data into tiny "micro-batches." This keeps all GPUs busy—as soon as GPU A finishes the first micro-batch and sends it to GPU B, it immediately starts on the second one.


## 2. Tensor Parallelism (Intra-layer Splitting)
This is a much "deeper" slice. Instead of splitting by layers, we split a **single mathematical operation** (like a massive Matrix Multiplication) across multiple GPUs.

* **The Workflow:** Imagine a layer that multiplies a 10,000x10,000 matrix. We split that matrix in half. **GPU A** calculates the top half of the result, and **GPU B** calculates the bottom half.
* **The Advantage:** This is the only way to handle individual layers that are too wide to fit in 192GB of VRAM (common in Ultra-LLMs).
* **The Requirement:** This requires ultra-fast, low-latency communication like **NVLink 5.0**, because the GPUs must "talk" to each other constantly during every single layer calculation, not just at the end.


## Comparison: Pipeline vs. Tensor Parallelism

| Feature | Pipeline Parallelism | Tensor Parallelism |
| :--- | :--- | :--- |
| **Slicing Method** | Between Layers (Vertical) | Inside Layers (Horizontal) |
| **Communication** | Low (Only between chunks) | **Very High** (Every layer) |
| **Network Needed** | Standard InfiniBand / Ethernet | **NVLink / Infinity Fabric** |
| **Complexity** | High (Requires scheduling) | Very High (Requires custom kernels) |


## 3. 3D Parallelism: The "Holy Grail"
For the largest models in 2026 (10T+ parameters), researchers use **3D Parallelism**, which combines all three techniques at once:
1.  **Data Parallelism:** To handle massive amounts of text.
2.  **Pipeline Parallelism:** To stack hundreds of layers across different server racks.
3.  **Tensor Parallelism:** To split individual wide layers across GPUs within a single rack.
