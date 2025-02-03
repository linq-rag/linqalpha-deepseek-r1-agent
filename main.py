# main.py
import argparse
from config import safe_log
from agent import Agent
from schemas.output_schema import build_dynamic_output_schema
from prompt.template import make_system_prompt
from tools.loader import load_tools_from_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the DeepSeek R1 Agent")
    parser.add_argument("--tools-config", default="tools_config.json", 
                        help="Path to the JSON file defining function metadata & executor mapping.")
    parser.add_argument("--query", default="What was the impact on stock prices due to DeepSeek's AI model release, and what are the societal implications?",
                        help="User query for the agent to process.")
    args = parser.parse_args()

    # Dynamically load function metadata & executors
    tools_metadata, tools_executor = load_tools_from_json(args.tools_config)

    # Build a system prompt from the loaded metadata
    system_prompt = make_system_prompt(tools_metadata)
    safe_log(system_prompt, "SYSTEM_PROMPT")

    # Build the output schema from the same metadata
    output_schema = build_dynamic_output_schema(tools_metadata)

    # Create the agent
    agent = Agent(
        system_prompt=system_prompt,
        query=args.query,
        tools_metadata=tools_metadata,
        tools_executor=tools_executor,
        output_schema=output_schema,
        max_iterations=6
    )

    # Run the agent
    reasoning, structured_output = agent.run()
    safe_log("Final Reasoning:", "COMPLETE")
    safe_log(reasoning)
    safe_log("Final Output:", "COMPLETE")
    safe_log(structured_output.answer)
