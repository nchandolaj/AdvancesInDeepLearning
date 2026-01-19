# Training Large Models: Memory Requirements

**Memory requirements**
* **Without optimizations**
  - Model parameters: N
  - Weights: N floats
  - Gradients: N floats
  - 1st momentum term: N floats
  - 2nd momentum term (Adam / AdamW) : N floats
  - **TOTAL:** **4 N parameters or floats**. That's **16 N bytes** without counting activations (because each float requires 4 bytes of memory)
* **With optimizations**
  - **1-2N bytes**

For contrast, 
* A small LLM with N = 8 Billion parameters (say, **Llama 3 with 8 Billion parameters**), requires **4N parameters** that includes weights, gradient, first & second momentum terms (refer to 'Memory Requirements' above).
* Each float requires 4 bytes of memory. So, for N = 8 Billion parameters, we need 32 GB of memory. And the entire small LLM with weigths, gradients, momentum terms, requires **128 GB of memory**.  
* An advanced GPU, say H100 chip has 80 GB HBM3 memory. Essentially, there is no GPU out right now that would fit the small LLM.

## How can we reduce memory requirements by making training more memory efficient?

* Mixed precision training
* Distributed Training
* Zero redundancy training
* Low-rank adapters
* Quantization
* Quantized Low-rank adapters
* Low-rank projections
* Checkpointing
* FlashAttention
* Open-source Infrastrucxture for model training

