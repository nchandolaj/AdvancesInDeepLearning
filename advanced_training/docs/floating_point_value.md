# Floating Point Value

A **floating point value** is a computer data type used to represent real numbers (numbers with fractional parts). 
It is based on **scientific notation**, allowing the binary "decimal point" to "float" to represent a vast range of magnitudes—from the microscopic width of an atom to the cosmic distances between galaxies.

In modern computing, these are almost universally standardized by the **IEEE 754** specification.

## 1. Key Sub-Components
A floating point number is physically divided into three distinct segments in memory:

| Component | Bit Size (Single) | Signifies... |
| :--- | :--- | :--- |
| **Sign Bit** | 1 bit | The polarity of the number. `0` = positive, `1` = negative. |
| **Exponent** | 8 bits | The **scale** or magnitude. It tells you how far to move the decimal point. |
| **Significand** / **Mantissa** | 23 bits | The **precision** or significant digits. It holds the actual "meat" of the value. |

### A. The Sign Bit
The most significant bit (leftmost). It determines if the value is above or below zero. Interestingly, this allows for two representations of zero: **+0** and **-0**.

### B. The Exponent (The Scale)
This part is stored as a **biased integer**. For a 32-bit float, a bias of **127** is added to the actual exponent.
* **Why bias?** It allows the exponent to be stored as an unsigned number (0 to 255) while still representing both very large positive and very small negative powers without needing a separate sign bit for the exponent itself.
* **Mathematical Shift:** If the stored exponent is 130, the "real" exponent used in calculation is $130 - 127 = 3$.

### C. The Significand / Mantissa (The Precision)
This represents the fractional part of the number. 
* **The Hidden Bit:** To save space, the leading `1.` in binary scientific notation (e.g., $1.101 \times 2^3$) is **not stored**. The hardware simply assumes it's there. This effectively gives you 24 bits of precision while only using 23 bits of memory.


## 2. Memory Structure
Memory-wise, a floating point is just a sequence of bits. The hardware interprets these bits using the following formula:

$$Value = (-1)^{Sign} \times (1.Significand) \times 2^{(Exponent - Bias)}$$

### Comparison of Formats
The more bits you give to the exponent, the larger your **range**. The more bits you give to the significand, the higher your **precision**.

| Format | Total Bits | Range (Approx) | Precision (Decimal Digits) |
| :--- | :--- | :--- | :--- |
| **Single (FP32)** | 32 | $10^{-38}$ to $10^{38}$ | ~7 digits |
| **Double (FP64)** | 64 | $10^{-308}$ to $10^{308}$ | ~16 digits |
| **Bfloat16** | 16 | Same as FP32 | ~3 digits |


### 3. Special Memory States
IEEE 754 reserves certain bit patterns in the exponent and significand for special conditions:
* **Infinity:** Exponent is all `1`s, Significand is all `0`s.
* **NaN (Not a Number):** Exponent is all `1`s, Significand is **not** `0`. (Occurs for $0/0$ or $\sqrt{-1}$).
* **Denormalized:** Exponent is all `0`s. Used to represent extremely tiny numbers near zero.

--- 

