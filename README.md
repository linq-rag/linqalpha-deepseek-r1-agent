# linqalpha-deepseek-r1-agent

**Authors**:  
- *[Suyeol Yun](https://github.com/syyunn), Fundamental Research Engineer @ [Linqalpha](https://www.linqalpha.com)*  
- *[Subeen Pang](https://www.linkedin.com/in/subeen-pang/), Fundamental Research Engineer @ [Linqalpha](https://www.linqalpha.com)*  
- *[Yongjin Kim](https://www.linkedin.com/in/yjin-kim/), Fundamental Research Engineer @ [Linqalpha](https://www.linqalpha.com)*  
- *[Chanyeol Jacob Choi](https://www.linkedin.com/in/chanyeolchoi/), CEO @ [Linqalpha](https://www.linqalpha.com)*  

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

## Parallel Function Calls & Reasoning Transparency

A key feature of this agent is its ability to **execute multiple function calls in parallel** within a single interaction. This design:

1. **Improves Efficiency**:  
   - Can gather data from multiple sources simultaneously
   - Reduces total processing time by avoiding sequential API calls
   - Allows cross-referencing of information from different tools

2. **Enhanced Reasoning Visibility**:  
   Each function call requires an explicit `reason` field explaining:
   - Why this specific function was chosen
   - How its arguments were determined
   - What information it's expected to provide

Example of parallel function calls with reasoning:
```json
{
  "function_calls": [
    {
      "name": "search_google",
      "arguments": {
        "query": "DeepSeek AI stock price impact",
        "start_date": "2024-01-01"
      },
      "reason": "To gather recent financial market reactions to DeepSeek's release"
    },
    {
      "name": "search_google",
      "arguments": {
        "query": "DeepSeek AI technical capabilities comparison",
        "start_date": null
      },
      "reason": "To understand technical differentiators affecting market perception"
    }
  ]
}
```

This combination of parallel execution and explicit reasoning helps the agent work more efficiently while maintaining full transparency in its decision-making process.

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

Below is a **suggested README section** demonstrating how users can **add or modify custom tools** using a JSON config (`tools_config.json`), without having to edit Python code directly. You can place it toward the end of your README—or wherever you think it best fits.

---

## Adding or Customizing Tools

You can define new functions (tools) or modify existing ones **without** editing `main.py` or other core files. We provide a `tools_config.json` file (or you can create your own) that follows [**OpenAI function-calling** conventions](https://platform.openai.com/docs/guides/function-calling#defining-functions). Each tool entry:

- Specifies the **Python module** and **function name** for the agent to call  
- Contains the **OpenAI-style** function metadata (`"type"`, `"function"`, `"description"`, `"parameters"`, etc.)

### 1. Implement Your Python Function

Create a Python module in the `tools/` directory (or anywhere else in your project). For example:

```python
# tools/my_custom_tool.py

def my_custom_tool(custom_arg: str) -> dict:
    """Example custom logic for demonstration purposes."""
    # Do something interesting here
    return {"result": f"Your argument was: {custom_arg}"}
```

### 2. Define It in `tools_config.json`

Edit (or create) a JSON file, e.g. `tools_config.json`:

```jsonc
{
  "executor_map": {
    // The key must match the function's "name" in your metadata.
    "my_custom_tool": {
      "module": "tools.my_custom_tool",
      "function_name": "my_custom_tool"
    }
  },
  "metadata": [
    {
      "type": "function",
      "function": {
        "name": "my_custom_tool",
        "description": "Runs a custom operation using 'custom_arg'.",
        "parameters": {
          "type": "object",
          "properties": {
            "custom_arg": {
              "type": "string",
              "description": "Any string to demonstrate how the tool is called."
            }
          },
          "required": ["custom_arg"],
          "additionalProperties": false
        }
      }
    }
  ]
}
```

Key fields:

1. **executor_map**: Tells the agent which Python module + function name to dynamically import for each tool.  
   - In this case, `"my_custom_tool"` loads `tools/my_custom_tool.py` and the function `my_custom_tool`.
2. **metadata**: An **array** of objects that conform to OpenAI’s “function calling” schema, including:
   - `"type": "function"`
   - A nested `"function"` object with `"name"`, `"description"`, and `"parameters"` (JSON schema).

### 3. Run the Agent with Your Tools

If you want to use this custom config file instead of the default `tools_config.json`, you can specify:

```bash
python main.py \
  --tools-config tools_config.json \
  --query "Demo how to use my_custom_tool, please!"
```

1. **The agent** will **import** your `my_custom_tool()` Python function,  
2. **Register** it under the name `"my_custom_tool"`,  
3. **Make it available** for the DeepSeek R1 model to call if/when it decides it’s relevant.

That’s all! The **only** changes required were:

1. Writing a Python function in `tools/`.
2. Adding a new block in your JSON config.

No changes to `main.py` or other parts of the codebase are needed.

---

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
    "function_calls":[
    {
      "name": "search_google",
      "arguments": {
        "query": "DeepSeek AI model release stock market impact 2024-2025",
        "start_date": "01/01/2024",
        "end_date": "02/02/2025"
      },
      "reason": "To find recent financial data and analyses on stock price changes linked to DeepSeek's AI release."
    },
    {
      "name": "search_google",
      "arguments": {
        "query": "societal implications of DeepSeek AI model ethical concerns automation 2024",
        "start_date": null,
        "end_date": null
      },
      "reason": "To gather authoritative sources discussing AI's broader societal impacts post-release."
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

Run `python main.py` yourself to see the full output and experiment with different queries or new tools!


