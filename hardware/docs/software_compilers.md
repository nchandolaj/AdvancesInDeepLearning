# Software Compilers (CUDA, ROCm, XLA)

When you write code in a high-level framework like PyTorch, JAX, or TensorFlow, it doesn't run directly on the silicon. Each vendor uses a distinct "compiler pipeline" to translate your math into the specific electrical signals (instructions) the hardware understands.

This page covers the following across the three major architectures (CUDA/ROCm, XLA):
* Translation of code / math to machine / hardware instructions
* "Matrix Multiply and Accumulate" (MAC) Operation
* "Mixed-Precision Training"

---

# Translation of Code / Math to Hardware Instructions
Here is how the software stack handles that translation across the three major architectures as of 2026.

## 1. NVIDIA: The CUDA & PTX Pipeline
NVIDIA uses a two-stage process that prioritizes **Forward Compatibility**. Your code is first turned into a "virtual" instruction set so it can run on future GPUs without being rewritten.

* **Stage 1: PTX (Parallel Thread Execution):** The compiler (NVCC) turns your C++/Python code into PTX, which is a low-level, human-readable "virtual" assembly.
* **Stage 2: SASS (Streaming Assembler):** At runtime, the NVIDIA driver takes that PTX and "jits" (Just-In-Time compiles) it into SASS. SASS is the true binary code that controls the Blackwell SMs.
* **Blackwell Specialization:** In the B200, the compiler now uses a new instruction set called **tcgen05**. Unlike older chips where all 32 threads in a Warp had to sync up to do a matrix multiply, `tcgen05` allows individual threads to trigger the Tensor Cores, reducing "idle time" and making the GPU more efficient.


## 2. AMD: The ROCm & LLVM Pipeline
AMD’s ROCm (Radeon Open Compute) stack is built on **LLVM**, an open-source compiler infrastructure. This makes it easier for developers to "peek under the hood."

* **HIP (Heterogeneous-Compute Interface for Portability):** Most AMD code is written in HIP. It looks almost exactly like CUDA. In fact, AMD provides a tool called "HIPIFY" that automatically converts NVIDIA code to AMD code.
* **Direct-to-ISA:** Unlike NVIDIA’s two-stage virtual approach, the ROCm compiler typically compiles code directly into the **GCN/CDNA ISA** (Instruction Set Architecture). 
* **MI355X Optimization:** For the newer CDNA 4 architecture, the compiler focuses on **"Kernel Fusion."** It looks at your math and "fuses" multiple steps into a single instruction to take advantage of the massive 288GB HBM3e memory, preventing data from ever having to leave the chip.


## 3. Google: The XLA & Systolic Dataflow
Google’s approach is fundamentally different because TPUs aren't general-purpose. They use a compiler called **XLA (Accelerated Linear Algebra)**.

* **Graph-Based Compilation:** XLA doesn't look at individual lines of code; it looks at the entire "Computation Graph." It sees the whole model at once.
* **Systolic Mapping:** Since the TPU v7 uses a **Systolic Array (MXU)**, the compiler’s job is to ensure a constant "heartbeat" of data. It arranges the data so that as it flows through the 256x256 grid, it hits every logic gate at exactly the right nanosecond. 
* **Ahead-of-Time (AOT):** While GPUs often decide what to do while the program is running, XLA does almost all the heavy lifting **before** the program starts. This is why TPUs are so power-efficient; they don't waste energy on "scheduling" or "branch prediction" during execution.


## Summary of Compilation Philosophies

| Feature | NVIDIA (Blackwell) | AMD (CDNA 4) | Google TPU (v7) |
| :--- | :--- | :--- | :--- |
| **Primary Compiler** | NVCC / CUDA | ROCm / LLVM | **XLA** |
| **Intermediate Rep** | **PTX** (Virtual Assembly) | LLVM IR | HLO (High-Level Optimizer) |
| **Scheduling** | Hardware-driven (Warp) | Hardware-driven (Wavefront) | **Compiler-driven** (Static) |
| **Key Advantage** | Runs on any NVIDIA GPU | Open-source and transparent | Maximum "Math-per-Watt" |

---

# "Matrix Multiply and Accumulate" (MAC) Operation

**Comparison of how a basic MAC operation is written in CUDA vs. how it is represented in XLA**

In 2026, the way hardware executes a **Matrix Multiply and Accumulate (MAC)** operation—the fundamental building block of AI—differs wildly between GPUs and TPUs.

The following examples demonstrate how **NVIDIA (CUDA)** focuses on managing threads and memory, while **Google (XLA/TPU)** focuses on a "dataflow graph" that the compiler maps to its physical systolic array.


## 1. NVIDIA (CUDA): Thread-Centric Matrix Multiply
In CUDA, you write a "Kernel" from the perspective of a single thread. You must manually calculate memory offsets and manage the movement of data between Global Memory and the fast L1/Shared Memory.

### CUDA C++ Example

CPP code
```cpp
// A naive CUDA kernel for C = A * B
__global__ void matrixMul(float* A, float* B, float* C, int N) {
    // Calculate the row and column index for this specific thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0;
        for (int k = 0; k < N; ++k) {
            // Manual memory indexing and accumulation
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}
```

* **The Hardware Logic:** This code launches thousands of threads. The hardware groups them into **Warps (32 threads)**. Each thread in the warp executes the `for` loop in lockstep.
* **Low-Level "PTX" Instruction:** At the silicon level, the Blackwell GPU uses a specific instruction called `mma.sync` (Matrix Multiply-Accumulate), which tells a Warp to collectively hand a small tile of data to the **Tensor Cores**.


## 2. Google (XLA): Graph-Centric Matrix Multiply
You rarely write "kernels" for a TPU. Instead, you write high-level math in a framework like JAX. The **XLA Compiler** then converts this into a "StableHLO" (High-Level Operation) graph.

### JAX / XLA Logic

Python code
```python
import jax.numpy as jnp
from jax import jit

@jit
def matmul_step(A, B):
    return jnp.dot(A, B) # The compiler sees this as a single 'dot' node
```

### The "Under the Hood" HLO (Intermediate Representation)
XLA transforms that `jnp.dot` into a declarative instruction like this:

Lisp code
```lisp
%dot.1 = f32[1024,1024] dot(f32[1024,512] %A, f32[512,1024] %B), 
         lhs_contracting_dims={1}, rhs_contracting_dims={0}
```

* **The Hardware Logic:** Unlike CUDA, which manages "who does what," XLA manages "**when** data arrives." 
* **The Systolic Heartbeat:** XLA schedules the data to flow into the **MXU (Matrix Multiply Unit)**. Because it’s a systolic array, the weights stay inside the logic gates, and the activations "pump" through them. There are no "threads" to manage; only a perfectly timed stream of data.


## Summary: How They Differ in Execution

| Concept | NVIDIA (CUDA) | Google (XLA / TPU) |
| :--- | :--- | :--- |
| **Control** | **Thread-based:** You tell threads where to go. | **Data-based:** You tell data when to move. |
| **Memory** | **Explicit:** You move data to "Shared Memory." | **Implicit:** The compiler handles all buffers. |
| **Optimization** | Done by the **Programmer** (or cuBLAS). | Done by the **Compiler** (XLA). |
| **Failure Mode** | "Race conditions" or memory leaks. | "Recompilation" or slow graph tracing. |

---

# Mixed-Precision Training

In 2026, the industry has shifted toward **Mixed-Precision Training**, where models are trained using ultra-low precisions like **FP8** and **FP4**. This allows for massive jumps in speed and memory efficiency without sacrificing the final accuracy of the model.

However, each vendor handles the "rounding" and "scaling" required for these low bits in fundamentally different ways.

## 1. NVIDIA Blackwell: The Microscaling Expert
NVIDIA Blackwell (B200) introduces the **2nd Generation Transformer Engine**, which specializes in **Microscaling** to prevent accuracy loss.

* **The Problem:** In a massive tensor, a few large "outlier" values can ruin the precision for all the small numbers if you use one scale factor for the whole tensor.
* **The Solution (NVFP4):** Instead of one scale for a whole tensor, Blackwell breaks the tensor into **micro-blocks of 16 elements**. Each block gets its own high-precision **E4M3 scale factor**.
* **Precision Support:** Blackwell supports a huge range: **FP4, FP6, FP8, INT8, BF16, FP16, TF32, FP32, and FP64**.
* **Scaling Strategy:** It uses **Current Scaling** (measuring the max value of the current batch) rather than **Delayed Scaling** (using the previous batch’s max).

## 2. AMD CDNA 4: The Open Standard Approach
AMD’s MI355X follows the **OCP (Open Compute Project)** standards for "Microscaling Formats" (MX).

* **Standardization:** AMD uses the community-defined **MXFP4** and **MXFP6**. This makes it easier to move models between different types of hardware (like moving a model from a TPU to an AMD GPU).
* **Performance:** By using MXFP4, the MI355X can hit a staggering **20.1 PetaFLOPS** of peak AI compute—roughly double its FP8 performance.
* **Hybrid Memory:** AMD’s massive **288GB HBM3e** memory allows it to keep the "Master Weights" in higher precision (like BF16) while doing the heavy math in FP4, preventing the model from becoming inaccurate during training.

## 3. Google TPU v7 (Ironwood): The Inference Era ASIC
The TPU v7 is Google's first chip designed specifically for the **"Age of Inference"** while maintaining world-class training capabilities.

* **Native FP8 Support:** Ironwood is the first TPU with native, hardware-level FP8 support. A single chip hits **4.6 PetaFLOPS** (FP8), which actually slightly exceeds the B200's raw FP8 throughput.
* **System-Level Co-Design:** Unlike GPUs that need the software to manage scaling, the TPU v7’s **XLA compiler** automatically decides where to inject FP8. It treats a cluster of 9,216 chips as a single supercomputer.
* **Power Efficiency:** By focusing on FP8, Google has achieved a **30x increase in performance-per-watt** compared to their original v1 TPU.

## High-Level Comparison: Mixed Precision (2026)

| Feature | **NVIDIA (Blackwell)** | **AMD (CDNA 4)** | **Google TPU (v7)** |
| :--- | :--- | :--- | :--- |
| **Flagship Precision** | **NVFP4** (4-bit) | **MXFP4** (4-bit) | **FP8** (8-bit) |
| **Scaling Granularity** | 16-element blocks | 32-element blocks | Tensor-level / Compiled |
| **Max AI Compute** | ~9 PetaFLOPS (FP4) | 20.1 PetaFLOPS (MXFP4) | 4.6 PetaFLOPS (FP8) |
| **Philosophy** | Custom & Proprietary | Open & Standardized | ASIC Efficiency |

### Logic Example: Microscaling
If you were to represent the scaling logic in a simplified pseudo-code, it would look like this:

```python
# Simplified Micro-block scaling logic
def quantize_to_nvfp4(tensor_block_16):
    # 1. Find the max value in this specific block of 16
    block_max = find_max(tensor_block_16)
    
    # 2. Store that max as a high-precision FP8 'scale'
    scale_factor = quantize_to_fp8(block_max)
    
    # 3. Scale all 16 values so they fit in the 4-bit range (-6 to 6)
    quantized_block = [val / scale_factor for val in tensor_block_16]
    return quantized_block, scale_factor
```
