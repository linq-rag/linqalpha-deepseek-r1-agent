# linqalpha-deepseek-r1-agent

**Author**: *[Suyeol Yun](https://github.com/syyunn), AI Agent Architect & Fundamental Research Engineer @ [Linqalpha](https://www.linqalpha.com)*

---

## Overview
**linqalpha-deepseek-r1-agent** demonstrates an **iterative function-calling agent** that uses [DeepSeek R1](https://github.com/deepseek-ai/DeepSeek-R1)—an advanced reasoning model—and relies on [Fireworks](https://fireworks.ai/models/fireworks/deepseek-r1) for **convenient inference and structured JSON output**. This design allows the agent to:

1. Generate **reasoning traces** to explain *why* it takes certain steps.  
2. Make **structured function calls** (e.g., web searches) when more information is needed.  
3. Self-correct or "heal" if partial failures arise (e.g., an error from a function call or JSON parse error).  

By handling intermediate answers, function calls, and errors in **strictly validated JSON** (via Pydantic), the agent can refine its output until it confidently reaches a final solution.

---

## What Is an Iterative Agent?
An **iterative agent**:
1. Takes a **user query** (e.g., "What is the impact on stock prices due to DeepSeek R1's release?").
2. **Reasons** about how to address it, possibly calling external "tools" (e.g., Google search, DB lookup).
3. **Interprets new data** from those tools to refine or correct its partial answer.
4. **Repeats** until confident in a final answer or until it hits a maximum iteration limit.

This approach emphasizes **transparency**, as the agent outputs a chain of reasoning steps. Developers or end users can trace how each partial result influenced the final outcome.

---

## Why Reasoning + Function Calls?
1. **Transparency**:  
   The chain-of-thought or "reasoning trace" helps reveal how the agent forms its conclusions—an important part of **interpretability**.  

2. **Debugging & Reliability**:  
   If the agent produces an incomplete or invalid response, you can review each reasoning step and identify *where* the logic went astray, then guide corrections.

3. **Human-Like Problem Solving**:  
   Iterative steps mirror how humans gradually refine ideas—incorporating or discarding evidence as more information becomes available.

**Note**: While much research on interpretability, such as [Anthropic's studies](https://www.anthropic.com/research/mapping-mind-language-model), often focuses on **model safety**, we find interpretability equally crucial for building stable, controllable, and **iteratively improvable** AI systems. Understanding *why* a model produces certain outputs can help ensure consistent, reliable behavior across daily tasks.

---

## Self-Healing Example
If the model outputs invalid or incomplete JSON, or if a function call fails (e.g., returns an error), that *error* is fed back into the agent's reasoning. The agent can then:
1. **Observe** the error message (e.g., "JSON parse error: missing field `query`").
2. **Generate** a revised output that corrects the JSON or modifies the function arguments accordingly.
3. **Continue** the iteration loop with the corrected data or function call.

This "self-healing" loop ensures the agent can recover gracefully from partial failures—rather than halting or returning a broken final result.

---

## Installation

Below is a **step-by-step** guide using **conda** to create a Python 3.10 environment:

1. **Clone or download** this repo:
   ```bash
   git clone https://github.com/linqalpha/linqalpha-deepseek-r1-agent.git
   cd linqalpha-deepseek-r1-agent
   ```

2. **Create a conda environment** (Python ≥ 3.10):
   ```bash
   conda create -n dpsk-r1-agent python=3.10
   conda activate dpsk-r1-agent
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   # Copy the example environment file
   cp .env-example .env
   ```
   
   Then open `.env` and add your API keys:
   ```ini
   FIREWORKS_API_KEY=fw-your-fireworks-api-key-here
   SERPAPI_API_KEY=your-serpapi-api-key-here
   ```

## Usage

Once your environment is set up:

1. **Activate your environment** if not already:
   ```bash
   conda activate dpsk-r1-agent
   ```

2. **Run the main script**:
   ```bash
   python main.py
   ```

The script will:
- Load your system prompt and user prompt
- Call DeepSeek R1 (via Fireworks) to process the query
- Call `search_google`  tools if additional data is needed
- Show any errors or incomplete JSON in logs (visible to users and fed back to the model for self-correction)


## Running Example

Below is a **condensed output** from running `main.py`. It shows how the **DeepSeek R1** model (accessed through **Fireworks**) iteratively reasons about a question, calls external functions, and generates a final answer with citations.

<details>
<summary>Example Agent Interaction</summary>

```
[SYSTEM_PROMPT] ... (System instructions and schema requirements) ...

[AGENT] Starting agent with query: "What was the impact on stock prices due to DeepSeek's AI model release, and what are the societal implications?"

[ITERATION] 1/6 
[API] Sending request to DeepSeek API... 
[API] DeepSeek API response time: 23.71 seconds

DeepSeek reasoning: 
"We should gather recent data from 2024-2025 to verify stock market reactions after the model release ... We'll need to call 'search_google' with relevant queries."

Structured output: 
{
    "keep_going": true,
    "reason_for_keep_going": "Initial analysis requires verification...",
    "answer": "Preliminary analysis suggests DeepSeek's AI model boosted investor sentiment...",
    "function_calls": [
        {
            "name": "search_google",
            "arguments": {
                "query": "DeepSeek AI model release stock price impact 2024-2025",
                "start_date": "01/01/2024",
                "end_date": "12/31/2024"
            }
        }
    ]
}

[TOOLS] Processing function calls... 
→ Calling 'search_google' with query="DeepSeek AI model release stock price impact 2024-2025"

[TOOL CALL RESULT] (Receives search results from SerpAPI)

[ITERATION] 2/6
[API] Sending request to DeepSeek API...
[API] DeepSeek API response time: 53.48 seconds

DeepSeek reasoning:
"We see multiple articles referencing a market plunge in late January 2025, plus bans by the U.S. Navy and Italy over security concerns..."

Structured output:
{
    "keep_going": false,
    "answer": "DeepSeek's AI model release in late 2024 led to mixed stock market impacts and significant societal concerns. While some stocks saw short-term gains, U.S. tech faced pressure amid geopolitical tensions...",
    "summary_reasoning": "...",
    "function_calls": null
}

[COMPLETE] Final Output:
DeepSeek's AI model release in late 2024 led to mixed stock market impacts ... Semiconductor suppliers like SK Hynix benefited, with their HBM chips ... Societally, ethical and security concerns prompted institutional bans ...

Key Sources:
- [2025-01-27] Chinese AI App DeepSeek Rattles Tech Markets
- [2025-01-28] U.S. Navy bans use of DeepSeek
- [2025-01-30] Italy bans DeepSeek AI ...
```
</details>

### Key Takeaways from the Example

1. **Iterative Reasoning**: The model first proposes a partial answer, then calls functions (e.g., Google search) to validate or expand its knowledge.  
2. **Self-Healing**: If any errors appeared (like missing JSON fields), the agent could correct them in the next iteration.  
3. **Rich Sources**: The final answer includes links to the most relevant articles discovered during the search phase, complete with context and citations.

Run `python main.py` yourself to see the full output and experiment with different queries or new tools!