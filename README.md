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