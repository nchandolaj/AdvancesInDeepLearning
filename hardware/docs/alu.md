# Arithmetic Logic Unit

An **ALU (Arithmetic Logic Unit)** is the "calculator" of the computer. It is the specific part of a processor that performs all mathematical and logical operations. While the CPU acts as the brain, the ALU is the specific circuit within that brain doing the actual "crunching."


## 1. What does an ALU do?
The ALU performs two main categories of tasks:

* **Arithmetic Operations:** It handles basic integer math, such as **Addition**, **Subtraction**, and sometimes **Multiplication**.
* **Logical Operations:** It performs "decision-making" math using logic gates. This includes **AND**, **OR**, **NOT**, and **XOR** operations.


## 2. Anatomy of an ALU
To perform a calculation, the ALU requires three main things:
1. **Operands (Input):** The actual binary numbers to be processed.
2. **Opcode (Instruction):** A code telling the ALU *which* operation to perform.
3. **Status Flags (Output):** Info about the result (Zero, Carry, Negative, etc.).


## 3. ALU vs. CPU vs. GPU vs. TPU
| Component | Role of the ALU / Logic |
| :--- | :--- |
| **CPU** | A few complex ALUs for general logic and branching. |
| **GPU** | Thousands of simple ALUs for parallel math. |
| **TPU** | A **Systolic Array** that functions as one giant matrix-math engine. |


## 4. How it works (The Logic Gate level)
At the hardware level, an ALU is built from:
* **Logic Gates:** (AND, OR, NOT) for decision making.
* **Full Adders:** Specialized circuits for binary addition.
* **Multiplexers (MUX):** Used to select which operation's output is sent to the final result.

---

# Difference between ALU and FPU

While both are "calculators" inside a processor, the fundamental difference lies in **what** they calculate and **how** complex their circuitry is.


## 1. The Core Difference: Integers vs. Decimals
* **ALU (Arithmetic Logic Unit):** Specialist for **Integers** (whole numbers) and **Logic** (AND/OR, comparisons).
* **FPU (Floating-Point Unit):** Specialist for **Decimals** (fractions and scientific notation).


## 2. Comparison Table
| Feature | ALU (Integer Unit) | FPU (Decimal Unit) |
| :--- | :--- | :--- |
| **Data Type** | Integers (Whole Numbers) | Floating-Point (Decimals) |
| **Operations** | Add, Sub, AND, OR, bit-shifts | Add, Sub, Mul, Div, SqRoot, Sin/Cos |
| **Latency** | Very Fast (~1 cycle) | Slower (3–10+ cycles) |
| **Usage** | OS Logic, Memory Addresses | 3D Graphics, Physics, AI |


## 3. Hardware Implementation
* **Silicon Area:** An FPU is much "larger" (uses more transistors) than an ALU because it must handle exponents and mantissas.
* **Modern Design:** Most CPU cores contain multiple ALUs and fewer (but very wide) FPUs/SIMD units.
* **Software Emulation:** If a processor lacks an FPU, it can perform decimal math on the ALU using software libraries, but it is often 10x–100x slower.
