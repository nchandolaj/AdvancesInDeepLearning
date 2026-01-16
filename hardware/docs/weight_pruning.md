# Model Optimization: Weight Pruning

**Weight Pruning** is the second major pillar of model optimization (alongside Quantization). While quantization makes the "numbers" smaller, pruning makes the "network" smaller by literally deleting the parts of the model that aren't contributing to its intelligence.


## 1. The Logic: Brain Synapses vs. AI Weights
Think of a child's brain development. A toddler has a massive number of neural connections. As they grow and learn, the brain undergoes "synaptic pruning"—it kills off weak or unused connections to become faster and more efficient. AI pruning works exactly the same way.

In a typical large model (like GPT-4), millions of the "weights" (the numbers representing connections) are very close to zero. They are doing almost nothing, yet the GPU still has to spend energy and time multiplying them.


## 2. How it Works: The Pruning Workflow
1.  **Evaluate:** The system looks at every weight in the model.
2.  **Threshold:** It identifies weights that are below a certain value (e.g., any weight between $-0.001$ and $0.001$).
3.  **Zero Out:** It sets those weak weights to exactly **zero**.
4.  **Compress:** Because multiplying by zero always equals zero, we can "skip" these calculations entirely.


## 3. Two Types of Pruning
The way you delete connections affects how much faster the hardware can actually run:

### **A. Unstructured Pruning**
Weights are deleted randomly wherever they are small.
* **Result:** A "Swiss cheese" model.
* **Problem:** Most hardware (like standard GPUs) isn't very good at skipping "random" zeros. It still has to look at the empty space, so you save memory, but you might not save much speed.

### **B. Structured (Sparse) Pruning**
Weights are deleted in specific patterns (e.g., deleting an entire row or a $2 \times 4$ block of neurons).
* **Result:** Large chunks of the model are removed.
* **Benefit:** Modern chips like **NVIDIA Blackwell** have "Sparse Core" hardware that can physically skip these empty blocks, leading to a **2x speedup** in performance.



## 4. The "Golden Trio" of Optimization
In 2026, most models deployed on your phone or laptop use all three of these techniques at once:
1.  **Pruning:** Delete the useless connections (Reduce quantity).
2.  **Quantization:** Shrink the remaining connections to 8-bit or 4-bit (Reduce precision).
3.  **Distillation:** Use a large "Teacher" model to train a tiny "Student" model to mimic its behavior.

