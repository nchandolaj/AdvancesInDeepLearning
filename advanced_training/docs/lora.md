# LoRA: Low-Rank Adapter

Training massive models like GPT-4 or Llama 3 is computationally expensive. And requires many GPUs if we were to use techniques like ZeRO Redundancy Optimization. Low-Rank Adapters is a cheaper alternative.

**LoRA (Low-Rank Adaptation)** is a popular **Parameter-Efficient Fine-Tuning (PEFT)** technique that allows you to **adapt large models to specific tasks without updating all their billions of parameters**.

Instead of retraining the entire weight matrix, LoRA freezes the original weights and **injects small, trainable "adapter" matrices** into the model.


## 1. The Core Concept: Weight Decomposition
In a standard neural network layer, you have a weight matrix $W$ of size $d \times k$. When fine-tuning normally, you update every single value in $W$.

LoRA assumes that the change in weights ($\Delta W$) during task-specific adaptation has a **"low intrinsic rank."** This means you don't need to update every parameter to get the same effect. LoRA represents the update as the product of two smaller matrices, $A$ and $B$:

$$\Delta W = B \times A$$

* **Original Matrix ($W$):** $d \times k$ (Frozen)
* **Matrix $A$:** $r \times k$ (Trainable)
* **Matrix $B$:** $d \times r$ (Trainable)
* **Rank ($r$):** A very small number (e.g., 4, 8, or 16).


## 2. The LoRA Workflow

### Step 1: Freeze the Backbone
The original pre-trained weights ($W_0$) are set to "non-trainable." This ensures the model retains its broad general knowledge (preventing **catastrophic forgetting**).

### Step 2: Inject Adapters
The two small matrices ($A$ and $B$) are added alongside the original layer. 
* **Matrix $A$** is typically initialized with a Gaussian distribution (random noise).
* **Matrix $B$** is initialized to zero, ensuring that at the very start of training, the "adapter" has zero effect ($B \times A = 0$).


### Step 3: Forward Pass
During training and inference, the input $x$ is passed through both the frozen weights and the trainable adapters. The results are summed:
$$h = W_0x + \Delta Wx = W_0x + BAx$$

### Step 4: Backpropagation
Gradients are calculated only for matrices $A$ and $B$. Since $r$ is much smaller than $d$ or $k$, the number of parameters being updated is often **less than 1%** of the original model size.


## 3. Why use LoRA?

| Feature | Full Fine-Tuning | LoRA |
| :--- | :--- | :--- |
| **Trainable Parameters** | 100% | < 1% |
| **VRAM Usage** | Very High | Low |
| **Storage Requirement** | Gigabytes (full model) | Megabytes (just adapters) |
| **Inference Latency** | High | **Zero** (can be merged) |


## 4. The "Weight Merging" Advantage
One of LoRA's coolest features is that it adds **zero latency** during inference. 

Because the update is just a matrix addition, you can mathematically merge the trained weights back into the original model once training is done:
$$W_{new} = W_0 + (B \times A)$$
After merging, you have a single matrix again. You get the specialized performance of the fine-tuned model without any extra computational overhead during the actual use of the model.


## 5. Summary of Benefits
* **Hardware Accessibility:** You can fine-tune a 70B parameter model on consumer-grade GPUs (like a 3090 or 4090) using LoRA (or **QLoRA**, which uses 4-bit quantization).
* **Portability:** You can share your "adapter" as a small file (10MB–100MB) rather than the entire 140GB model.
* **Switchability:** A single server can host one base model and swap different LoRA adapters in and out instantly to handle different tasks (e.g., one for coding, one for creative writing).


