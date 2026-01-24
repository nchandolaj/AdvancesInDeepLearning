# Training: Low-Rank Adapters

Clever machine learning trick to optimize fewer weights in your network. Especially helpful if you are not GPU-rich, i.e., you don't have a lot of GPUs to use ZeRO / FSDP.

Problem we are trying to solve is "memory-wall".

During training, the following takes up memory:
* Model weights
* Gradients
* Momentum terms
* Activations - *Intermediate outputs that we use to compute the forward of an MLP.*

Our focus here is **gradients** and **momentum terms** 

**Idea: Train fewer parameters**
* Keep most parameters frozen
  * No gradient, no momentum
* Train a small subset (of parameters left)
  * With gradient and momentum

How do we select the parameters?

## Fine-Tuning Classifier

Add a classifier layer

* Freeze backbone
* Train classifier
* Most memory efficient
  * No backdrop
  * Very few learnable parameters
* Not very expressive (*this is an issue*)

We still need to go forward throurh the network, but we only go back a layer or so 

Forward is often faster than backwards in a network.

## Input Adapters

Since a Fine-Tuning Classifier is not very expressive, a second way to fine tune is to fine tune both inputs and outputs of the model.

So, we have a classifier in the end, we are also learning the first few layers.

Add a classifier layer

* Freeze backbone
* Train input embedding (maybe classifier)
* Fairly memory efficient
  * Very few learnable parameters
* Popular with LLMs
  * Soft-prompting
  * Adapters for new inputs

We need to do backpropagation through the entire network.

## Intermediate layers

* Fine-tuning input and and output does not change computation inside network significantly
  * Cannot learn new "internal computation"
* Can we learn a subset of intermediate layer parameters?

## Low Rank Adapters (LoRA)

A better way to learn a subset of weights. 

* Keep weights W of the network is (N by M) matrices 
* Low Rank Adapter: Learn adapter AB to replace W
  * A is (N by R), B is (R x M)
  * Rank R << min (M, N); R is much smaller than M and N
  * It just means that the amount of information captured in AB is limited.
* Total Parameters: R (N + M)

How do we train the model W?
* LoRAs always require a pre-trained model.

How do we initialize A and B?
* **A** small random (normal)
* **B** zero

**LORA models**
* Most weights frozen
* Train adapter for
  * All linear layers
  * Just MLPs
  * Just Attention
* Optionally train full input and output embedding

**Memory Requirements**
Lora
* Model parameters: N, LoRA param M
* Weights: N + M
* Gradients: M floats
* Momentum: M floats
* 2nd Momentum (ADAM): M floats
* **TOTAL:** 4 N + 16 M bytes without activations
* **M often ~ 1-5% of N** 

