# Multi-Architecture Illustration: The Hierarchical Chat

To illustrate a Multi-Agent Architecture, I will use the most common pattern: **The Hierarchical Chat** (often called the "Manager-Worker" or "Boss-Employee" pattern).

In this structure, a "Manager" agent breaks down a complex goal and delegates tasks to specialized "Worker" agents who have specific tools and skills.

## The Scenario: "Build a Stock Price Scraper"
Imagine a user gives a vague instruction: *"Create a Python script that scrapes the current price of Apple stock and saves it to a CSV file every hour."*

Here is how a 3-Agent Swarm handles this.

## The Cast of Agents

1.  **The Manager (Orchestrator):**
    * **Role:** Holds the "state" (the big picture). It never writes code or runs tools. Its job is to plan, delegate, and review.
    * **System Prompt:** "You are a Senior Product Manager. Your goal is to oversee the creation of a Python script. Break the user's request into steps. Assign steps to the Developer. Review their work. If the work is good, pass it to the QA. If it fails, send it back to the Developer."

2.  **Worker A: The Developer (Specialist):**
    * **Role:** Writes code and executes it.
    * **Tools:** Python Interpreter, File System access.
    * **System Prompt:** "You are a Senior Python Developer. Write efficient code based on the Manager's instructions. Use the 'Run Code' tool to test your work before replying."

3.  **Worker B: The QA Engineer (Critic):**
    * **Role:** Reviews code for bugs, security risks, or logic errors.
    * **System Prompt:** "You are a QA Engineer. Critique the code provided. Look for infinite loops, missing error handling, or security vulnerabilities. If you find issues, reject the work."

## The Workflow: Step-by-Step

This entire conversation happens autonomously in the background.

### Step 1: The Plan (User -> Manager)
* **User:** "Build a stock scraper for Apple."
* **Manager (Internal Thought):** *I need to break this down. 1. Write the scraper logic. 2. Add the scheduling logic. 3. specific library requests.*
* **Manager (to Developer):** "Please write a Python function that fetches the current price of 'AAPL' using the `yfinance` library. Test it to make sure it returns a number."

### Step 2: Execution (Manager -> Developer)
* **Developer:** Receives the task. It writes the code block.
* **Developer (Tool Use):** *Calls Python Interpreter.*
    ```python
    import yfinance as yf
    stock = yf.Ticker("AAPL")
    print(stock.history(period='1d')['Close'][0])
    ```
* **Tool Output:** `Error: No module named 'yfinance'`
* **Developer (Self-Correction):** *Ah, I need to install it first or handle the import error.* (Refines code).
* **Developer (to Manager):** "I have successfully created the function. It fetches the price. Here is the code..."

### Step 3: Review (Manager -> QA)
* **Manager (to QA):** "The Developer has submitted this code. Please review it for safety and errors."
* **QA:** Scans the code.
* **QA (to Manager):** "REJECT. The code works, but it lacks error handling. If the internet is down, the script will crash. Tell the Developer to wrap the request in a `try-except` block."

### Step 4: Iteration (Manager -> Developer)
* **Manager (to Developer):** "Good start, but the QA found an issue. You are missing error handling for network failures. Please refactor."
* **Developer:** Rewrites the code with `try-except`.
* **Developer (to Manager):** "Fixed. Here is the robust version."

### Step 5: Completion (Manager -> User)
* **Manager:** Determines the plan is complete.
* **Manager (to User):** "Here is your finished Python script. It includes error handling and uses the yfinance library."


## Why use this complex structure?

You might ask, "Why not just ask GPT-4 to do it all in one go?"

1.  **Context Window Management:** If the task is massive (e.g., "Build a whole website"), a single model will "forget" the beginning of the code by the time it reaches the end. By breaking it up, the Developer agent only needs to focus on *one function at a time*.
2.  **Specialization:** You can give the "QA Agent" specific rules (e.g., "Always check for GDPR compliance") that you don't want to clutter the Developer's instructions with.
3.  **Self-Correction:** The "Manager" acts as a loop-breaker. If the Developer gets stuck in a loop, the Manager can see it happening and say, "Stop. Try a different approach," preventing the agent from wasting money on infinite retries.
