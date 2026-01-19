# Training Large Models

Machine Learning models started evolving with the advancements in the hardware (GPUs) that they used.

## History

* **Pre 2012:**
  - Small CPUs only (Intel Core i3/i5), convex optimization, limited performance.
  - Limited to Convex optimization.
  - Hand engineered features, small datasets
  - Limitations: Human engineering, CPU compute
* **2012-2018:**
  - Single GPU models (GTX 1080), non-convex optimizers (Adam, better versions of SGD), better architectures (network structures), large datasets
  - Limitations: GPU compute
* **2019-2022:**
  - Multi-GPU models, multi-dataset models
  - Transformer Architecture (Attention-based models),
  - LAION 5B Dataset
  - Limitations: GPU compute + **memory**
* **2023-Present:**
  - Frontier models (massive models - 8GB-400B parameters),
  - Multi-Node training / AI Clusters / Nodes of GPUs / Data Centers with many Nodes
  - Internet-scale datasets - Petabytes of Web Archives
  - Limitations: **GPU memory**

--- 

## 2012-Present: A Deeper Dive through a different Lens
The history of machine learning shifted radically around 2010, moving from the era of "feature engineering" (where humans manually defined patterns for computers to find) to "representation learning" (where computers learned the patterns themselves).

Here is the brief history of this evolution, broken down by the key eras and the specific factors that set them apart.


### 1. The Deep Learning Big Bang (2012–2015)
The tipping point occurred in 2012. Before this, neural networks were seen as computationally too expensive and ineffective compared to simpler statistical methods.

* **The Turning Point:** A model called **AlexNet** decimated the competition at the ImageNet Large Scale Visual Recognition Challenge in 2012, reducing the error rate from ~26% to 15.3%.
* **Architecture Wise:** This era was defined by **Convolutional Neural Networks (CNNs)**. Unlike previous models that flattened images into simple lists of numbers, CNNs preserved spatial relationships (understanding that a pixel is related to the pixels around it). Deeper models like **VGG** and **GoogleNet** followed, proving that increasing "depth" (number of layers) significantly improved intelligence.
* **Hardware Wise:** This was the birth of **GPU computing** for AI. Researchers realized that Graphics Processing Units (NVIDIA GTX 580s), originally designed for video games, could perform the matrix calculations needed for neural networks much faster than CPUs.
* **Dataset Wise:** The **ImageNet** dataset (14 million labeled images) was the fuel. It proved that deep learning algorithms required massive data to generalize well, unlike earlier models trained on small, hand-picked sets.


### 2. The Era of Depth and Generative AI (2015–2017)
As models grew deeper, they became harder to train (the "vanishing gradient" problem).

* **Architecture Wise:** The introduction of **ResNet (Residual Networks)** in 2015 solved the depth problem by using "skip connections," allowing data to bypass certain layers. This allowed networks to go from ~20 layers to hundreds or thousands without breaking. Simultaneously, **GANs (Generative Adversarial Networks)** introduced the concept of two networks fighting each other—one creating fake data, the other detecting it—which birthed the early era of AI image generation.
* **Performance Wise:** Models began exceeding human-level performance in specific narrow tasks, such as image classification and board games (AlphaGo in 2016).


### 3. The Transformer Revolution (2017–Present)
In 2017, Google researchers published "Attention Is All You Need," proposing the **Transformer** architecture. This is the foundation of modern Generative AI (ChatGPT, Claude, Gemini).

* **Architecture Wise:** Transformers replaced Recurrent Neural Networks (RNNs). Old RNNs read text sequentially (left to right), meaning they often "forgot" the beginning of a long sentence by the time they reached the end. Transformers read the entire sequence at once using a mechanism called **Self-Attention**, allowing the model to understand the relationship between every word and every other word simultaneously.
* **Model Wise:** This shift moved the field toward **Foundation Models** (like BERT, GPT). Instead of training a new model for every single task (one for translation, one for summary), we now train one massive model on *everything* and fine-tune it for specific tasks.
* **Hardware Wise:** The rise of **TPUs (Tensor Processing Units)** and massive GPU clusters (H100s). Training these models requires interconnecting thousands of chips to act as a single supercomputer.
* **Dataset Wise:** We moved from labeled datasets (like ImageNet) to **web-scale unsupervised data**. Models are now fed seemingly the entire internet (Common Crawl, books, code repositories) and learn by predicting the next word, rather than being told explicit answers.


### Summary of What Set Them Apart

| Dimension | Pre-2010 (Traditional ML) | 2010–2017 (Deep Learning) | 2017–Present (GenAI) |
| :--- | :--- | :--- | :--- |
| **Model Focus** | SVMs, Random Forests, Logistic Regression | CNNs (Vision), RNNs/LSTMs (Text) | Transformers (GPT, PaLM, Llama) |
| **Architecture** | Shallow (flat math functions) | Deep (Layers of neurons) | Parallel (Attention mechanisms) |
| **Input Data** | Structured data (Excel sheets, small tables) | Perceptual data (Images, Audio) | Multimodal (Text + Code + Image + Video) |
| **Hardware** | CPUs | Single/Multi GPUs | Distributed GPU/TPU Clusters |
| **Key Differentiator** | **Feature Engineering:** Humans had to tell the model *what* to look for. | **Representation Learning:** The model learned features, but task-specific. | **Generalization:** One model can do many tasks without specific training. |

