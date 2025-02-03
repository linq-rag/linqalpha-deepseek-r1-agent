# main.py
from config import safe_log
from agent import Agent
from tools.search import SEARCH_GOOGLE_TOOL_METADATA, search_google
from schemas.output_schema import build_dynamic_output_schema
from prompt.template import make_system_prompt

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    MAX_ITERATIONS = 6
    tools_metadata = [SEARCH_GOOGLE_TOOL_METADATA] # append more tools here as needed
    tools_executor = {
        "search_google": search_google # append more tools here as needed. the key should match the name of the tool in the tools_metadata
    }   
    # Generate system prompt using the new function
    system_prompt = make_system_prompt(tools_metadata)
    safe_log(system_prompt, "SYSTEM_PROMPT")
    output_schema = build_dynamic_output_schema(tools_metadata)

    query="What was the impact on stock prices due to DeepSeek's AI model release, and what are the societal implications?",

    agent = Agent(
        system_prompt=system_prompt,
        query=query,
        tools_metadata=tools_metadata,
        tools_executor=tools_executor,
        output_schema=output_schema,
        max_iterations=MAX_ITERATIONS
    )
    
    reasoning, structured_output = agent.run()
    safe_log("Final Reasoning:", "COMPLETE")
    safe_log(reasoning)
    safe_log("Final Output:", "COMPLETE")
    safe_log(structured_output.answer)