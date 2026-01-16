# Floating-Point Unit (FPU)

In a CPU, the **Floating-Point Unit (FPU)** is a specialized hardware component designed to perform math on real numbers (decimals like $3.14159$ or $0.000123$) rather than whole integers ($1$, $2$, $3$). 

While the concept of "doing math with decimals" is the same across CPUs, GPUs, and TPUs, the **way** the hardware is built and **how** it processes that math is fundamentally different.


## 1. The CPU FPU: The "Artisan"
In a CPU, the FPU is built for **precision and complexity**. It is an "artisan" that handles one very difficult calculation at a time with extreme reliability.

* **Role:** Handles complex scientific, engineering, and financial math that requires "Double Precision" ($FP64$) or higher.
* **Design:** Each CPU core typically has one or two FPUs. If you have an 8-core CPU, you have 8-16 FPUs.
* **Flexibility:** It can do a wide variety of tasks: addition, division, square roots, and trigonometric functions (sine, cosine).
* **Standard:** It strictly follows the **IEEE 754** standard to ensure that $2.0 + 2.0$ always equals $4.0000000000000000$ exactly, every single time, across any software.


## 2. The GPU FPU: The "Factory Line"
In a GPU, floating-point math is handled by **thousands** of tiny, simplified units (often called CUDA Cores or Stream Processors).

* **Role:** Handles "vector" math. Instead of one complex equation, it does the same simple math to thousands of pixels or data points simultaneously.
* **Design:** A modern GPU like the **H100** doesn't have 8 FPUs; it has **thousands** ($18,000+$). 
* **Specialization:** Most GPU FPUs are optimized for $FP32$ (Single Precision) or $FP16$ (Half Precision), which is "good enough" for graphics and AI but less precise than a CPU's $FP64$.
* **Comparison:** If a CPU FPU is a master chef cooking one gourmet meal, a GPU is a massive fast-food assembly line making 10,000 burgers at once.


## 3. The TPU: The "Matrix Engine"
A TPU (Tensor Processing Unit) doesn't really have a "Floating-Point Unit" in the traditional sense. It uses something called a **Systolic Array** (the Matrix Multiply Unit or MXU).

* **Role:** It is designed for one specific type of floating-point math: **Matrix Multiplication**.
* **Design:** While a CPU and GPU "load" numbers, "calculate" them, and "save" them back to memory, a TPU lets data flow through a giant grid of logic gates like a heartbeat (systolic). 
* **Innovation:** TPUs pioneered formats like **bfloat16** (Brain Floating Point), which has the same range as a CPU's $FP32$ but uses half the memory. It is less precise for scientific simulations but perfect for AI.


## Summary Comparison

| Feature | CPU (FPU) | GPU (ALU/Core) | TPU (MXU) |
| :--- | :--- | :--- | :--- |
| **Quantity** | Very Few (2–64) | Thousands (1,000+) | One or two massive arrays |
| **Philosophy** | **Scalar:** 1 + 1 = 2 | **Vector:** [1,2] + [3,4] | **Tensor:** [Matrix] × [Matrix] |
| **Precision** | High ($FP64$) | Medium ($FP32/FP16$) | Low/Optimized ($BF16/FP8$) |
| **Best For** | Operating systems, logic | Graphics, general AI | Large-scale AI training |
