# Multi-Agent Architecture: The Manager-Work-QA Implementation

Here is an actual Python implementation of the **Manager-Worker-QA** structure that uses **The Hierarchical Chat** pattern. This is an example of **Multi-Agent Architecture**.

I have chosen **Microsoft AutoGen** for this example because it is currently the industry standard for conversational "multi-agent swarms." It handles the turn-taking (who speaks next) and the message history automatically.

### The AutoGen Implementation [DRAFT]

In this code, we create a **Group Chat**. The "GroupChatManager" acts as the central brain that decides which agent speaks next based on the conversation history.

```python
import autogen

# 1. Configuration: Tell the agents which LLM to use (e.g., GPT-4 or Claude 3.5)
llm_config = {
    "config_list": [{"model": "gpt-4", "api_key": "YOUR_API_KEY"}],
    "temperature": 0.2, # Low temperature for more precise, less creative code
}

# 2. Define the User Proxy (The "Client")
# This agent acts as the bridge between YOU (the human) and the agents.
# It executes the code tools if human_input_mode is "NEVER" or asks you for permission.
user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="A human admin. Execute the code provided by the Coder.",
    code_execution_config={"work_dir": "coding", "use_docker": False},
    human_input_mode="TERMINATE", # Asks for human input only when tasks are done
)

# 3. Define the Manager (The "Planner")
# This agent breaks down the task but does not write code.
pm_agent = autogen.AssistantAgent(
    name="Product_Manager",
    llm_config=llm_config,
    system_message="""
    You are a Senior Product Manager. 
    1. Analyze the user's request.
    2. Create a step-by-step plan for the Coder.
    3. Review the Coder's output. If the QA Engineer rejects it, ask the Coder to fix it.
    4. Only when the QA Engineer approves, reply with 'TERMINATE'.
    """
)

# 4. Define the Worker (The "Coder")
coder_agent = autogen.AssistantAgent(
    name="Coder",
    llm_config=llm_config,
    system_message="""
    You are a Senior Python Developer.
    1. Write code to solve the tasks assigned by the Product Manager.
    2. Wrap your code in markdown blocks.
    3. If the code fails or is rejected, fix it and output the new version.
    """
)

# 5. Define the Critic (The "QA")
qa_agent = autogen.AssistantAgent(
    name="QA_Engineer",
    llm_config=llm_config,
    system_message="""
    You are a QA Engineer.
    1. Review the code written by the Coder.
    2. Check for: Bugs, Security issues, and Missing error handling.
    3. If the code is bad, reply with 'REJECT' and explain why.
    4. If the code is good, reply with 'APPROVE'.
    """
)

# 6. Create the "Room" (GroupChat)
# This logic controls the 'Speaker Selection' (who talks next).
group_chat = autogen.GroupChat(
    agents=[user_proxy, pm_agent, coder_agent, qa_agent], 
    messages=[], 
    max_round=12  # Prevents infinite loops if they argue too long
)

# 7. Create the Manager Object
# This is the actual AI model that runs the group chat logic.
manager = autogen.GroupChatManager(groupchat=group_chat, llm_config=llm_config)

# 8. Start the Process
user_proxy.initiate_chat(
    manager,
    message="Build a Python script that scrapes the current stock price of Apple (AAPL) using yfinance and saves it to a CSV."
)
```

### How this Code Runs

When you run this script, you will see a console log that looks like a screenplay:

1.  **Admin (UserProxy):** Sends the prompt "Build a Python script..."
2.  **Product_Manager:** Intercepts. "Okay, Coder, first install yfinance, then write the script..."
3.  **Coder:** Generates the code block.
4.  **Admin (UserProxy):** *Automatically detects the code block and runs it locally.* It returns the output (or error message) into the chat.
5.  **QA_Engineer:** Reads the output. "Wait, you didn't handle the case where the API is down. REJECT."
6.  **Product_Manager:** "Coder, please fix the error handling."
7.  **Coder:** Writes v2 of the code.
8.  **QA_Engineer:** "APPROVE."
9.  **Product_Manager:** "TERMINATE."

### Alternative: LangChain (LangGraph)
While AutoGen is great for "conversations," **LangGraph** is better if you want strict control (e.g., *always* go from Coder $\to$ QA, never let Coder talk to PM directly).

Here is a snippet of how that logic looks in LangGraph:

```python
from langgraph.graph import StateGraph, END

# Define the flow explicitly
workflow = StateGraph(AgentState)

# Add nodes (Agents)
workflow.add_node("planner", planner_agent)
workflow.add_node("coder", coder_agent)
workflow.add_node("qa", qa_agent)

# Add edges (The rigid rules of movement)
workflow.add_edge("planner", "coder")
workflow.add_edge("coder", "qa")

# Conditional Edge: If QA says "reject", go back to Coder. Else, End.
workflow.add_conditional_edges(
    "qa",
    should_continue, # A python function that checks for "APPROVE" string
    {
        "continue": "coder",
        "end": END
    }
)

app = workflow.compile()
```

### Which one should you choose?

* **Use AutoGen** if you want **Creative Problem Solving**. Because agents can "choose" who speaks next, they might surprise you (e.g., the Coder might ask the PM a clarifying question before writing code).
* **Use LangGraph** if you want **Reliable Production Pipelines**. If you need to guarantee that the QA step *never* gets skipped, a graph is safer than a conversation.

