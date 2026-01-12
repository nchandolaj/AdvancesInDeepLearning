# Software Compilers (CUDA, ROCm, XLA)

When you write code in a high-level framework like PyTorch, JAX, or TensorFlow, it doesn't run directly on the silicon. Each vendor uses a distinct "compiler pipeline" to translate your math into the specific electrical signals (instructions) the hardware understands.

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

# "Matrix Multiply and Accumulate" (MAC)Operation

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
