# Mixed Precision Training

Mixed-precision training is a technique used to accelerate the training of deep neural networks by using different numerical formats for different parts of the computation. 
Instead of performing every calculation in high-precision **FP32** (32-bit floating point), it uses lower-precision formats like **FP16** or **BF16** (16-bit) for the bulk of the math, 
while maintaining a master copy of the weights in FP32 to ensure accuracy.

**Advantages:** 
* Reduce memory used in training deep learning
* Train model faster. Traditionally, we use floating point values to train deep networks.

## 1. What problem is it solving?
Deep Learning models are becoming massive (hundreds of billions of parameters). Training them in full FP32 creates three primary bottlenecks:

* **Memory Wall:** Large models simply don't fit into GPU memory (VRAM). **FP32 requires 4 bytes per parameter**; reducing this to 2 bytes (FP16/BF16) or 1 byte (FP8) effectively doubles or quadruples the capacity of the hardware.
* **Bandwidth Bottleneck:** Moving 32-bit numbers between the GPU memory and the processing cores is slow. Smaller numbers travel faster through the "pipes."
* **Compute Bottleneck:** Modern AI hardware (like NVIDIA's Tensor Cores) is physically built to perform 16-bit or 8-bit operations much faster than 32-bit operations (often 8x to 16x faster).

## 2. The Underlying Memory Structure
In mixed-precision training, the memory is managed through a **Master Weight** system. 

1.  **The FP32 Master Copy:** A full-precision version of the weights is kept in memory.
2.  **The FP16 Copy:** During the "Forward Pass" (calculating the guess) and the "Backward Pass" (calculating the error), a 16-bit version of the weights is used.
3.  **The Update:** Once the errors (gradients) are calculated, they are applied to the **FP32 Master Copy**.

## 3. Mathematical Explanation & Advantages

### Numerical Representation
To understand the advantage, we look at the IEEE 754 standard for floating-point numbers. A number is represented as:
$$V = (-1)^s \times M \times 2^E$$
* **s:** Sign bit.
* **M:** Mantissa (precision/significant digits).
* **E:** Exponent (dynamic range).

| Format | Bits (Total) | Exponent Bits (Range) | Mantissa Bits (Precision) |
| :--- | :--- | :--- | :--- |
| **FP32** | 32 | 8 | 23 |
| **FP16** | 16 | 5 | 10 |
| **BF16** | 16 | 8 | 7 |

### The Advantages
1.  **Dynamic Range vs. Precision:** **BF16** (Brain Floating Point) is often preferred over FP16 because it keeps the same 8-bit exponent as FP32. This means it can represent the same range of very large and very small numbers, even if it is less "precise" about the exact digits.
2.  **Loss Scaling:** Because FP16 has a narrow range, small gradients can "underflow" to zero. Mathematically, we solve this by multiplying the loss by a factor $S$ (e.g., 1024) before backpropagation:
   
    $$\text{Scaled Gradient} = \nabla(Loss \times S)$$

     This pushes the values into the representable range of the 16-bit format.

### Float 32 / Float16 / (B)float16 (See Appendix)
* (B)float16 uses half the memory
* (B)float16 can be more than twice as fast -
  - partly because of memory efficiency and
  - partly because of algorithm
  - most operations are memory bound
  - some operations have super-linear memory access patterns

FLOPs: Floating Point Operations

## 4. Are we sacrificing anything?
While highly effective, there are trade-offs:

* **Numerical Stability:** If not managed correctly (without loss scaling), gradients can vanish to zero or explode to infinity ($NaN$), causing the training to fail.
* **Memory Overhead:** Paradoxically, you actually use *more* memory for a single parameter during training because you store both the FP16 version and the FP32 master version. However, the savings in "Activations" (the data stored between layers) are so massive that the overall memory footprint still drops significantly.
* **Implementation Complexity:** It requires careful hyperparameter tuning, though modern frameworks like PyTorch (`torch.cuda.amp`) and JAX handle most of this automatically today.


## 5. Training with BF16

### Computation in bfloat16 - Case Study: matmul

C = A.B

Matrix multiplications are implemented in block form on GPUs, where slices of data (of A and B, from the main memory of the the GPU) are loaded into the shared memory of the GPU, 
which reduces the number of times we have to read the original matrices (A and B) from the main memory of the GPU.

Each read is half size
* 2 x faster

Block size b^2 can be 2 x larger
* square root of 2 * faster

**Total: 2.82 x faster**. In practicew, there are other considerations, so the nbr isnt exactly 2.82 times faster.

### Loss of precision (underflows)
**Solution:** Gradient Scaling
* Multiple loss by a large value (i.e. 2^16)
* If overflow (inf or NaN)
  - Ignore gradient (set to 0)
  - Lower gradient scale

### Adam in (b)float16

* First momentum: possible
* Second momentum: Unstable, Not recommended. So, keep in 32 bit if you can.
* Specialized very low-bit Adam implementations exist 

## 6. Summary

**Memory Requirements w/ Mixed Precision**
* Weight (fp32): 4N bytes
* Gradient (Bf16): 2N bytes
* 1st Momentum (Bf16): 2N bytes
* 2nd Momentum (fp32): 4N bytes

Thats **12N bytes** without counting activations. Reduced from **16N bytes** without optimizations.

---
# PyTorch AMP Implementation

Here is how the theoretical "Master Weight" system and "Loss Scaling" are implemented in code using PyTorch's **Automatic Mixed Precision (AMP)** library.

## 1. The Implementation Logic
PyTorch handles the complexity via two main components:
* **`torch.cuda.amp.autocast`**: A context manager that automatically handles the "Casting" (moving data from FP32 to FP16) only for operations where it is safe and beneficial (like Matrix Multiplications).
* **`torch.cuda.amp.GradScaler`**: A utility that manages "Loss Scaling" to prevent gradients from flushing to zero (underflow).

### PyTorch AMP Code Example
```python
import torch
from torch.cuda.amp import autocast, GradScaler

# 1. Initialize model and optimizer in FP32 (Default)
# These are your 'Master Weights'
model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 2. Create a GradScaler to handle Dynamic Loss Scaling
scaler = GradScaler()

for inputs, labels in data_loader:
    optimizer.zero_grad()

    # 3. Forward Pass with Autocast
    # Temporarily casts weights to FP16 for fast Tensor Core math
    with autocast():
        output = model(inputs)
        loss = criterion(output, labels)

    # 4. Scale the Loss and Backward Pass
    # Multiplies loss by a large factor (e.g., 65536) to lift gradients above zero
    scaler.scale(loss).backward()

    # 5. Unscale gradients and update weights
    # scaler.step() first unscales the gradients back to FP32.
    # If it detects 'NaN' or 'Inf' (overflow), it skips this update.
    scaler.step(optimizer)

    # 6. Update the scale factor for the next iteration
    # If an overflow occurred, it decreases the scale; otherwise, it stays or grows.
    scaler.update()
```

## 2. What is happening under the hood?
When you use the code above, the hardware follows a specific data flow to protect the precision of your model while maximizing speed.

### Step-by-Step Data Flow
1.  **Casting:** The FP32 "Master Weights" remain in memory, but a temporary FP16 version is created for the forward pass.
2.  **FP16 Forward Pass:** The GPU's Tensor Cores perform the bulk of the math using these 16-bit weights and activations.
3.  **Scaling:** The resulting loss is multiplied by the `GradScaler`'s current factor.
4.  **FP16 Backward Pass:** Gradients are calculated in 16-bit. Because the loss was scaled, these gradients stay within the representable range of FP16.
5.  **Unscaling & Update:** * The `scaler` checks if any gradients are `Inf` or `NaN`. 
    * If the gradients are valid, they are converted back to FP32 and divided by the scale factor.
    * These high-precision "unscaled" gradients are then applied directly to the **FP32 Master Weights**.

## 3. Comparison of Training Modes

| Feature | FP32 Training (Full) | AMP (Mixed) | FP16 Training (Pure) |
| :--- | :--- | :--- | :--- |
| **Speed** | 1x (Baseline) | **~2x to 4x faster** | Fastest |
| **Memory Usage** | High | **Low (saves activations)** | Lowest |
| **Stability** | Highest | **High (due to Master Weights)** | Poor (high risk of $NaN$) |
| **Hardware** | Any GPU | **Tensor Core GPUs (Volta+)** | Tensor Core GPUs |

---
# APPENDIX

**float32** - **32-bit**
* Sign: **1 bit**
* Exponent: **8 bits**
* Mantissa (fraction): **23 bits**
* Precision (relative): 1E-07
* Max value: 3E+38
* Min value (normal): 1.7E-38

Precision (relative) = (x2 - x1) / x1 ; where x2 is smallest value x2 < x1

**float16** - **16-bit**
* Sign: **1 bit**
* Exponent: **5 bits**
* Mantissa (fraction): **10 bits**
* Precision (relative): 1E-04
* Max value: 65504
* Min value (normal): 6E-05

**Bfloat16** - **16-bit**
* Sign: **1 bit**
* Exponent: **8 bits**
* Mantissa (fraction): **7 bits**
* Precision (relative): 7.8E-03
* Max value: 3E+38
* Min value (normal): 1.7E-38
