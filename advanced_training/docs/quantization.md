# Quantization

In the world of Deep Learning, **Quantization** is essentially the art of "downsizing" a neural network. 

Training high-performance models usually requires high-precision math, but running those models on a phone or an IoT device is like trying to fit a grand piano into a studio apartment. Quantization solves this by reducing the precision of the model's weights and activations.


## 1. The Core Concept: From Floats to Ints
Most deep learning models are trained using **FP32** (32-bit Floating Point). This allows for extreme detail but takes up significant memory and power. Quantization converts these numbers into lower-bit formats, most commonly **INT8** (8-bit Integer).

* **FP32:** Can represent over 4 billion unique values.
* **INT8:** Can only represent 256 unique values (from -128 to 127).

By mapping the wide range of floating-point numbers into a smaller "bucket" of integers, you drastically reduce the model's footprint.


## 2. Why Do We Do It?
We make our data "less accurate," but it comes down to three major wins:

* **Memory Efficiency:** Moving from 32-bit to 8-bit cuts your model size by **4x**. This is the difference between an app being 500MB or 125MB.
* **Lower Latency:** Integers are much "cheaper" for a CPU or GPU to process than floating-point numbers. This means faster predictions (inference).
* **Power Savings:** Less data movement and simpler math mean your battery lasts longer. This is crucial for "Edge AI" (smartwatches, cameras, etc.).

## 3. Types of Quantization
There are two primary ways to implement this:

### Post-Training Quantization (PTQ)
This is the "easy" way. You take a fully trained model and convert it. It’s fast and requires very little data, but you might see a slight drop in accuracy because the model wasn't "warned" that its precision was about to be slashed.

### Quantization-Aware Training (QAT)
This is the "pro" way. During the training process, the model "simulates" the loss of precision. It learns to be robust against the rounding errors that happen during quantization. 
* **Result:** Usually maintains much higher accuracy than PTQ, especially for very small models.


## 4. The Trade-off: The "Quantization Error"
When you squeeze a continuous range of numbers into discrete steps, you introduce noise (Quantization Error). 

$$E = x - Q(x)$$

Where $x$ is the original value and $Q(x)$ is the quantized value. The goal of advanced quantization techniques is to minimize this error so the model's output remains virtually identical to the high-precision version.

---

# Discussion

## Memory Requirements

Just running the forward pass (e.g. inference) for model with N paramameters requires 4 N bytes.

Without optimization
* Model parameters : N
* Weights: N floats 
* 4 N bytes w/o counting activations

8 B parameter model (e.g. llama 3.1)
  * 32 GB  memory for just weights
  * So, even for inference alone, only a few advanced NVIDIA GPUs (A100, etc.) will fit the model.
  * Fast GPU memory HBM3 memory is quite expensive, and an engineering problem.
  * **bfloat16** (1-bit Sign, 8-bits Exponent, 7-bits Fraction)
  *   If we may use bfloat16 for weights, we can reduce memory requirements to say 16 GB.
    * With this loss in precision, we **lose fine-grained differences** between different values.
    * Also, for computation in bfloat16, we will have truncation (approximation) issue. While this may be an issue for computation, it is not a problem in storing the paramaeters on bfloat16.
    * So, how about going even lower... Float8
  * **float8** (1-bit Sign, 4-bits Exponent, 3-bits Fraction) 
    * Precision is limited (1.06 and 1.00 will look the same)
  * **float4** (1-bit Sign, 2-bits Exponent, 1-bit Fraction) 
    * We can't go to float4 for deep networks. It can store just a few values: 0.5, 1, 1.5, 2, 3, Inf, NaN, -ve of all these

### Integer Quantization
Instead opf using floats / precision, there is a better way to store values - **Integer Quantization** (2 types - Scale and Affline)

#### 1. Integer Scale Quantization

Quantization Error: T / (2^(k-1) - 1)

Weights: NK/8 + 2 bytes

#### 2. Integer Affine Quantization

Quantization Error: (B - A) / (2^k - 1) ; B is the smallest and A is the largest values of our range

Weights: NK/8 + 4 bytes


Both types of integer quantization are reliant on the largest possible value of your weights in your network. 

So, if any weight is absurdly large, for any reason, the entire quantization is messed up.
* This can be resolved by **Blockwise Quantization**

### Blockwise Quantization

Instead of quantizing the entire networks weights into the same fixed range, we compute **blocks of weights**.
And we store them together with the **quantized weights for each block**.

### Beyond Linear Quantization
There are other types of quantization techniques.

### 8-bit Adam

Quantize 1st and 2nd momentum in Adam
* 1st momentum term: int8
* 1st momentum term: uint8
* Non-linear quantization
Requires "stable" embeddings for LLMs
* 32-bit optimizer states, normalization

### Stochastic Rounding
How to train with quantized weights?
* Deterministic rounding
* Stochastic rounding

## How low can we go?

* GPT-style LLM can store about 2 bits of information per parameter
  * Under ideal conditions
* 4 bits in practice
  * Only after training!

## TLDR;
Training:
* Quantization helps reduce large memory requirements, say with 8-bit Adam, we need 6 N bytes (2N Weight bf16, 2N Gradient bf16, 1N Momentum int8 each, Total 6N bytes)
Inference:
* Quantization helps reduce large memory requirements, say for N parameters, we use 1/2 N bytes only.

