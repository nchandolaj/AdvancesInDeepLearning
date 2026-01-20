# Agentic AI


## What is Agentic AI?
**Agentic AI** refers to AI systems that can pursue complex goals with limited human supervision.

The easiest way to understand the shift is to compare it to the AI we've used for the last few years:
* **Generative AI (Passive):** You ask a chatbot to write an email. It writes the email. It stops.
* **Agentic AI (Active):** You tell an agent, "Plan and execute a marketing campaign for next week."
    * It *researches* trends on the web.
    * It *drafts* the emails.
    * It *logs in* to your CRM (like Salesforce) to find the right customers.
    * It *schedules* the sends.
    * It *monitors* the replies and adjusts strategy if open rates are low.

It is the shift from **"AI as a Tool"** (which you hold) to **"AI as a Teammate"** (which you manage).

## How It Works: The "Agentic Loop"
Standard LLMs just predict the next word. Agents wrap that LLM in a control loop that looks like this:

1.  **Perception:** The agent reads the user's goal ("Fix the bug in the login page").
2.  **Reasoning & Planning:** The agent breaks the goal down. *"First I need to reproduce the error, then I need to read the code, then write a test."*
3.  **Tool Use:** The agent has access to "arms and legs"—software functions it can trigger. It might use a **Browser** to search documentation, a **Code Interpreter** to run Python, or an **API** to send a Slack message.
4.  **Reflection:** After it acts, it observes the result. If its code fix fails, it reads the error message, realizes its mistake, and tries a different approach.

## Intended Use Cases (High Value)

The industry is moving away from "fun" demos to high-ROI enterprise applications.

### 1. Autonomous Software Engineering
This is currently the most mature sector for agents.
* **The Use Case:** Instead of a human developer spending 4 hours fixing a minor bug, an Agent (like **Devin** or **GitHub Copilot Workspace**) scans the repository, identifies the issue, writes the fix, runs the unit tests to ensure it didn't break anything else, and submits a Pull Request for human review.

### 2. Tier 1 Customer Support (Resolution, not just Chat)
* **The Use Case:** Old chatbots just deflected users to FAQs. Agentic support systems have permission to access backend databases.
* **Example:** A customer says, "My internet is down." The Agent checks the ISP's diagnostic tool, sees a router error, triggers a remote reset command, waits 2 minutes, pings the router to confirm it's back up, and then replies to the customer. Zero human intervention.

### 3. Data Analysis & Financial Auditing
* **The Use Case:** An agent is given a raw Excel dump of 10,000 transactions and told to "Find suspicious activity."
* **The Workflow:** The agent writes its own Python scripts to visualize the data, spots outliers, cross-references them with vendor invoices in a separate folder, and generates a PDF report summarizing the 5 most risky transactions.

### 4. "Sovereign" Research Agents
* **The Use Case:** Investment firms use agents to perform due diligence.
* **The Workflow:** "Research Company X." The agent scrapes their latest 10-K filings, reads recent news articles, checks employee sentiment on Glassdoor, and compiles a risk profile, citing every source it used.


## Industry Adoption Status (2026)

We are currently in the **"Orchestration" Phase**.

* **The Reality:** In 2023-2024, agents were often "toys"—they would get stuck in infinite loops or hallucinate wildly.
* **The Fix:** The industry solved this by moving from "One God Agent" to **"Multi-Agent Systems" (Swarms)**.
    * Instead of one AI trying to do everything, companies build a "Manager Agent" that delegates tasks to a "Coder Agent," a "Researcher Agent," and a "Reviewer Agent."
    * **Microsoft (AutoGen)**, **LangChain**, and **Salesforce (Agentforce)** are the leading frameworks enabling enterprises to build these specific swarms.

## Summary: The Market Shift

| Old Way (Generative AI) | New Way (Agentic AI) |
| :--- | :--- |
| **User Input:** "Write a SQL query for me." | **User Input:** "Update the database with this week's sales." |
| **Capability:** Text generation. | **Capability:** Tool execution (SQL, API, Browser). |
| **Success Metric:** Accuracy of text. | **Success Metric:** Successful completion of the task. |
| **Human Role:** Editor. | **Human Role:** Manager/Supervisor. |

