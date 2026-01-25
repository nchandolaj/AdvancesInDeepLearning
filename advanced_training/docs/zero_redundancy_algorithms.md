# ZeRO Algorithms: All-Reduce, Reduce Scatter, All-Gather


ZeRO Redundancy Optimization startegies use the following supporting algorithms:
* **Reduce-Scatter**
* **All-Gather**
* **All-Reduce**

---

## Algorithm Relationships

```mathematica
All Reduce = Reduce Scatter + All Gather
```

### Mental model

* **Reduce-Scatter**
  - Math plus partitioning

* **All-Gather**
  - Communication only

* **All-Reduce**
  - Everyone gets the same reduced result

### Terminologies Used

#### Rank 
A **rank** is a **unique numeric ID assigned to one participating worker** in a distributed job.

In **distributed training**, each process gets exactly **one rank**.

#### Rank vs GPU vs node
* **Rank** → communication identity
* **GPU** → compute device
* **Node** → physical machine

```
Node 0
  GPU 0 → Rank 0
  GPU 1 → Rank 1
  GPU 2 → Rank 2
  GPU 3 → Rank 3

Node 1
  GPU 0 → Rank 4
  GPU 1 → Rank 5
  GPU 2 → Rank 6
  GPU 3 → Rank 7
```

#### Shard
A **shard** is a contiguous slice of a larger tensor that has been partitioned across ranks so each worker owns only a fraction of the data.

Shard = 1 divided by number of ranks of a tensor

**Where shards appear in training systems**
| System	| What is sharded |
| :-- | :-- |
| DDP	| Nothing permanent |
| ZeRO-1	| Gradients and optimizer states |
| ZeRO-2	| Gradients, optimizer states, parameters during backward |
| ZeRO-3	| Everything |

#### Tensor

The **full tensor** is the whole book, and a **shard** is one chapter.

A **shard** is the piece of a **tensor** that a rank permanently keeps after distributed reduction, enabling large memory savings without changing the math.

---

## Reduce-Scatter: Reduce then Shard the Results
```mermaid
flowchart TB
    subgraph Input Gradients
        R0[Rank 0<br/>Shard 0 1 2 3]
        R1[Rank 1<br/>Shard 0 1 2 3]
        R2[Rank 2<br/>Shard 0 1 2 3]
        R3[Rank 3<br/>Shard 0 1 2 3]
    end

    R0 --> RS[Reduce Scatter<br/>Sum then partition]
    R1 --> RS
    R2 --> RS
    R3 --> RS

    RS --> O0[Rank 0<br/>Final Shard 0]
    RS --> O1[Rank 1<br/>Final Shard 1]
    RS --> O2[Rank 2<br/>Final Shard 2]
    RS --> O3[Rank 3<br/>Final Shard 3]
```

### Interpretation
* Each rank contributes a full tensor
* Reduction happens across ranks
* Output is **one reduced shard per rank**

---

## All-Gather: Shard to Full-Tensor
```mermaid
flowchart TB
    subgraph Input Shards
        S0[Rank 0<br/>Shard 0]
        S1[Rank 1<br/>Shard 1]
        S2[Rank 2<br/>Shard 2]
        S3[Rank 3<br/>Shard 3]
    end

    S0 --> AG[All Gather]
    S1 --> AG
    S2 --> AG
    S3 --> AG

    AG --> F0[Rank 0<br/>Full Tensor]
    AG --> F1[Rank 1<br/>Full Tensor]
    AG --> F2[Rank 2<br/>Full Tensor]
    AG --> F3[Rank 3<br/>Full Tensor]
```

### Interpretation
* No math happens
* Pure communication
* Everyone reconstructs the full tensor

---

## All-Reduce via both: Reduce-Scatter plus All-Gather
```mermaid
flowchart LR
    A[Full Tensor<br/>Each Rank] --> B[Reduce Scatter]
    B --> C[Shard per Rank]
    C --> D[All Gather]
    D --> E[Full Reduced Tensor<br/>Each Rank]
```

---

# Animated as a ring algorithm

## Ring Reduce-Scatter — step-by-step

Each step sends one shard clockwise, reduces on receipt, and discards what’s no longer needed.

### Step 1
```mermaid
sequenceDiagram
    participant R0 as Rank 0
    participant R1 as Rank 1
    participant R2 as Rank 2
    participant R3 as Rank 3

    R0->>R1: Shard 0
    R1->>R2: Shard 1
    R2->>R3: Shard 2
    R3->>R0: Shard 3
```

### Step 2
```mermaid
sequenceDiagram
    participant R0 as Rank 0
    participant R1 as Rank 1
    participant R2 as Rank 2
    participant R3 as Rank 3

    R0->>R1: Reduced Shard 3
    R1->>R2: Reduced Shard 0
    R2->>R3: Reduced Shard 1
    R3->>R0: Reduced Shard 2
```

### Step 3: Final Reduce-Scatter state
```mermaid
flowchart LR
    R0[Rank 0<br/>Final Shard 0]
    R1[Rank 1<br/>Final Shard 1]
    R2[Rank 2<br/>Final Shard 2]
    R3[Rank 3<br/>Final Shard 3]
```
### Result
* Each rank owns one fully reduced shard
* No rank ever materializes the full tensor

## Ring All-Gather — rebuilding the full tensor
Now the shards circulate again, but no reduction, only forwarding.

### Step 1
```mermaid
sequenceDiagram
    participant R0 as Rank 0
    participant R1 as Rank 1
    participant R2 as Rank 2
    participant R3 as Rank 3

    R0->>R1: Shard 0
    R1->>R2: Shard 1
    R2->>R3: Shard 2
    R3->>R0: Shard 3
```

### Step 2
```mermaid
sequenceDiagram
    participant R0 as Rank 0
    participant R1 as Rank 1
    participant R2 as Rank 2
    participant R3 as Rank 3

    R0->>R1: Shard 3
    R1->>R2: Shard 0
    R2->>R3: Shard 1
    R3->>R0: Shard 2
```

### Final State
```mermaid
flowchart LR
    F0[Rank 0<br/>Full Tensor]
    F1[Rank 1<br/>Full Tensor]
    F2[Rank 2<br/>Full Tensor]
    F3[Rank 3<br/>Full Tensor]
```

## Full ring All-Reduce summary

```mermaid
flowchart LR
    A[Full Tensor<br/>Each Rank]
    A --> B[Ring Reduce Scatter]
    B --> C[One Reduced Shard per Rank]
    C --> D[Ring All Gather]
    D --> E[Full Reduced Tensor<br/>Each Rank]
```

## Why the ring is powerful

* Bandwidth optimal
* Each link sends equal-sized chunks
* Total data per rank ≈ 2 times tensor size
* Scales linearly with number of ranks

---

# Unified Diagram with ZeRO-1/2/3 strategies

### Extended Reduce-Scatter context (pre-steps + backward)

**Canonical ZeRO-1 flow**

```mermaid
flowchart TD
    A[Mini Batch Input] --> B[Forward Pass<br/>Compute activations]

    B --> C[Backward Pass Starts<br/>Layer by layer]

    C --> D[Compute Local Gradients<br/>Full tensor per rank]

    D --> E[Partition Gradients<br/>Into equal shards]

    E --> F[Reduce Scatter<br/>Sum shards across ranks]

    F --> G[Each Rank Owns<br/>One Reduced Gradient Shard]

    G --> H[Optimizer Step<br/>Update owned parameters]
```

#### ZeRO-1 (only optimizer state is sharded)

**ZeRO 1**
Forward full → Backward full → Reduce scatter once

**Key point**
* Gradients exist in full briefly
* Reduce-scatter happens after gradient computation

```mermaid
flowchart TD
    P[Full Parameters<br/>Replicated] --> B[Forward Pass]
    B --> C[Backward Pass]
    C --> D[Full Gradients<br/>Per Rank]
    D --> E[Reduce Scatter]
    E --> O[Shard of Optimizer State<br/>Updated]
```

#### ZeRO-2 (gradients and optimizer state sharded)

**ZeRO 2**
Forward full → Backward layer → Reduce scatter per layer

**Key point**
* Reduce-scatter happens **layer by layer**
* Memory is freed earlier than ZeRO-1

```mermaid
flowchart TD
    P[Full Parameters<br/>Replicated] --> B[Forward Pass]

    B --> C[Backward Pass<br/>Layer i]

    C --> D[Gradient for Layer i]
    D --> E[Reduce Scatter<br/>Layer i]

    E --> F[Free Gradient Buffer<br/>Early]
```

#### ZeRO-3 (everything sharded)

**ZeRO-3**
Gather params → Compute → Reduce scatter → Free

**Key point**
* Parameters are gathered **just in time**
* Reduce-scatter immediately follows gradient computation
* Maximum memory efficiency

```mermaid
flowchart TD
    A[Parameter Shards<br/>Per Rank]
    A --> B[All Gather Params<br/>Layer i]

    B --> C[Forward Pass<br/>Layer i]

    C --> D[Backward Pass<br/>Layer i]

    D --> E[Reduce Scatter<br/>Gradients Layer i]

    E --> F[Release Params<br/>And Gradients]
```

---
