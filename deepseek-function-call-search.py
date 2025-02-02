import re
import os
import sys
import time
import json
import requests
import logging
from datetime import datetime
from typing import Any, Dict, List
from pydantic import BaseModel
from urllib.parse import urlencode

from openai import OpenAI
from utils import build_dynamic_output_schema


# -----------------------------
# Environment and Logging Setup
# -----------------------------
from dotenv import load_dotenv
load_dotenv()

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not FIREWORKS_API_KEY:
    raise ValueError("Missing FIREWORKS_API_KEY environment variable")
if not SERPAPI_API_KEY:
    raise ValueError("Missing SERPAPI_API_KEY environment variable")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def safe_log(message, prefix=None):
    """Log messages with consistent formatting and optional prefix."""
    try:
        formatted_message = ""
        if prefix:
            formatted_message = f"[{prefix}] "
        formatted_message += str(message)
        logger.info(formatted_message)
    except Exception as e:
        print(f"Logging failed: {e} - original message: {message}", file=sys.stderr)

# -----------------------------
# Define Search Functions (Tools)
# -----------------------------
def search_google(query, start_date=None, end_date=None, num_results=50):
    if not query:
        return {'error': 'query is required'}

    base_url = 'https://serpapi.com/search.json'
    params = {
        'engine': 'google',
        'q': query,
        'api_key': SERPAPI_API_KEY
    }

    # If both start_date and end_date are provided, construct tbs parameter
    if start_date and end_date:
        # Example format: tbs=cdr:1,cd_min:01/01/2022,cd_max:12/31/2022
        tbs_value = f"cdr:1,cd_min:{start_date},cd_max:{end_date}"
        params['tbs'] = tbs_value

    search_url = f"{base_url}?{urlencode(params)}"

    try:
        response = requests.get(search_url)
        response.raise_for_status()
        search_results = response.json()

        if not search_results.get('organic_results'):
            return {'error': 'No results found'}

        # Process and limit results
        limited_results = search_results['organic_results'][:num_results]
        formatted_results = [{
            'title': result.get('title'),
            'link': result.get('link'),
            'snippet': result.get('snippet'),
            'date': result.get('date', 'Date unavailable').replace(', +0000 UTC', '')
        } for result in limited_results]

        return formatted_results

    except requests.exceptions.RequestException as e:
        return {'error': f'SerpAPI request error: {str(e)}'}
    except Exception as e:
        return {'error': f'Unexpected error: {str(e)}'}

# -----------------------------
# Define Tools Metadata and Executor Mapping
# -----------------------------
tools_metadata = [   # Define google search tool
    {
        "type": "function",
        "function": {
            "name": "search_google",
            "description": "Search Google for a given query, returning top results. Optionally specify a date range.",
            "parameters": {
            "type": "object",
            "properties": {
                "query": {
                "type": "string",
                "description": "The search query."
                },
                "start_date": {
                "type": "string",
                "description": "The start date for filtering search results (MM/DD/YYYY). If omitted, date filtering won't be applied."
                },
                "end_date": {
                "type": "string",
                "description": "The end date for filtering search results (MM/DD/YYYY). If omitted, date filtering won't be applied."
                }
            },
            "required": ["query"],
            "additionalProperties": False
            }
        }
    }
]

tools_executor = {
    "search_google": search_google,
}

# -----------------------------
# Define the DeepSeek Chat Completion Function (Main Thinker)
# -----------------------------

def parse_deepseek_response(content: str, model_schema: type[BaseModel]) -> tuple[str, BaseModel]:
    """
    Parse DeepSeek response content into reasoning and model data.
    
    Args:
        content: Raw response content containing <think> tags and JSON
        model_schema: Pydantic model class for parsing the JSON data
        
    Returns:
        Tuple of (reasoning string, parsed model instance)
    """
    # Extract reasoning from <think> tags
    reasoning_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided."
    
    # Extract and parse JSON data
    json_match = re.search(r"</think>\s*(\{.*\})", content, re.DOTALL)
    json_str = json_match.group(1).strip() if json_match else "{}"
    model_data = model_schema.model_validate_json(json_str)
    
    return reasoning, model_data

def get_deepseek_response(messages: List[Dict[str, Any]], output_schema: type[BaseModel], model: str = "accounts/fireworks/models/deepseek-r1") -> tuple[str, BaseModel]:
    client = OpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=FIREWORKS_API_KEY,
    )

    start_time = time.time()
    safe_log("Sending request to DeepSeek API...", "API")
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object", "schema": output_schema.model_json_schema()},
        messages=[{"role": "user", "content": str(messages)}],
    )
    end_time = time.time()
    safe_log(f"DeepSeek API response time: {end_time - start_time:.2f} seconds", "API")
    
    return parse_deepseek_response(response.choices[0].message.content, output_schema)

# -----------------------------
# Define the Agent that Uses DeepSeek as Main Thinker
# -----------------------------
class Agent:
    def __init__(self, 
                 system_prompt: str, 
                 query: str,
                 tools_metadata: List[Dict[str, Any]], 
                 tools_executor: Dict[str, Any],
                 output_schema: type[BaseModel],
                 max_iterations: int = 6):
        self.system_prompt = system_prompt
        self.query = query
        self.tools_metadata = tools_metadata
        self.tools_executor = tools_executor
        self.output_schema = output_schema
        self.max_iterations = max_iterations

    def run(self) -> tuple[str, BaseModel]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.query}
        ]
        
        safe_log(f"Starting agent with query: {self.query}", "AGENT")
        
        for iteration in range(self.max_iterations):
            safe_log(f"\n{'='*50}")
            safe_log(f"Starting iteration {iteration + 1}/{self.max_iterations}", "ITERATION")
            
            try:
                reasoning, structured_output = get_deepseek_response(
                    messages=messages,
                    output_schema=self.output_schema,
                )

                safe_log("DeepSeek reasoning:", "REASONING")
                safe_log(reasoning)
                safe_log("Structured output:", "OUTPUT")
                safe_log(structured_output.model_dump_json(indent=2))

                if structured_output.function_calls:
                    safe_log("Processing function calls...", "TOOLS")
                    tool_calls_results = []
                    # Create a set of unique function calls based on name and arguments
                    seen_calls = set()
                    unique_calls = []
                    if structured_output.function_calls is not None:
                        for tool_call in structured_output.function_calls.root:
                            # Create a unique key from the function name and arguments
                            call_key = (
                                str(tool_call.name.value),
                                json.dumps(tool_call.arguments.dict(), sort_keys=True)
                            )
                            if call_key not in seen_calls:
                                seen_calls.add(call_key)
                                unique_calls.append(tool_call)
                        
                        # Process only unique function calls
                        for tool_call in unique_calls:
                            func_name = str(tool_call.name.value)  # Convert enum to string directly
                            if func_name in self.tools_executor:
                                safe_log(f"Calling function '{func_name}' with args: {tool_call.arguments.model_dump_json(indent=2)}")
                                tool_func = self.tools_executor[func_name]
                                args_dict = {k: v for k, v in tool_call.arguments.model_dump().items()}  # Changed from dict() to model_dump()
                                tool_result = tool_func(**args_dict)
                                safe_log(f"Result from '{func_name}': {json.dumps(tool_result, indent=2)}")
                                
                                tool_calls_results.append({
                                    "function": {
                                        "name": func_name,
                                        "arguments": json.dumps(args_dict)
                                    },
                                    "result": tool_result
                                })
                        
                        # Add single message with all function calls and results
                        messages.append({
                            "role": "This is the output from the function calls that you've requested.",
                            "content": json.dumps([call["result"] for call in tool_calls_results]),
                            "tool_calls": tool_calls_results
                        })

                        safe_log("Finished function calls.. resulsts will be submitted to DeepSeek")
                
                if not structured_output.keep_going:
                    safe_log("Stopping iterations: ", "COMPLETE")
                    safe_log(structured_output.reason_for_keep_going)
                    return reasoning, structured_output
                
                safe_log("Continuing to next iteration...", "ITERATION")
                    
            except Exception as e:
                safe_log(f"Error in iteration {iteration + 1}: {e}", "ERROR")
                messages.append({
                    "role": "error response",
                    "content": f"Error processing your output: {e}. Please reflect on this error and adjust your response accordingly to avoid this issue."
                })
        
        safe_log(f"Hit maximum iterations ({self.max_iterations})", "COMPLETE")
        return reasoning, structured_output

# -----------------------------
# Define the System Prompt
# -----------------------------
def make_system_prompt(tools_metadata: List[Dict[str, Any]], current_date: str = None, current_time: str = None) -> str:
    """Creates a system prompt that includes function descriptions and usage guidelines."""
    if current_date is None:
        current_date = datetime.now().strftime("%Y-%m-%d")
    if current_time is None:
        current_time = datetime.now().strftime("%H:%M:%S")

    # Extract function descriptions
    function_descriptions = []
    for tool in tools_metadata:
        if tool["type"] == "function":
            func = tool["function"]
            desc = (
                f"- {func['name']}: {func['description']}\n"
                f"  Arguments: {json.dumps(func['parameters']['properties'], indent=2)}"
            )
            function_descriptions.append(desc)

    # Create the base template without f-strings
    template = """
### Your Identity
You are a rational function-calling agent powered by DeepSeek (via Fireworks API).
Your task is to provide accurate answers by effectively utilizing the available functions.

### You're an Iterative Reasoner
You'll be provided with the initial user's question with your previous resposne to it, with the log of function calls and results also with your reasonings. 
If you want to streaming your reasnoing and response over and over again to make the final answer more accurate, please set the "keep_going" field to true and provide a detailed reasoning for why you want to continue.

### Available Functions:
{functions}

### Function Usage Guidelines:
1. Use the search_google function to gather relevant information from the web
2. Set appropriate date ranges for searches based on:
   - Context from previous answers
   - Timing of events in the query
   - Recent developments in the topic
3. Process search results to provide comprehensive answers

### Output Schema Requirements:
Your response must follow this JSON structure:
{{
    "keep_going": boolean,  /* indicates whether to continue processing */
    "reason_for_keep_going": string,  /* explanation of why processing should continue */
    "answer": string,  /* intermediate or final answer to track progress so far */
    "summary_reasoning": string,  /* summary of the reasoning process */
    "function_calls": [    /* list of function calls should be provided if keep_going is true */
        {{
            "name": string,  /* function name */
            "arguments": {{   /* function parameters */
                /* specific arguments for the function */
            }}
        }}
    ]
}}

Without these fields, your response will be invalid and cause an error. Function calls are good to be None if you don't need to use them.

Example Output:
{{
    "keep_going": true,
    "reason_for_keep_going": "Need to gather recent information about the topic",
    "answer": "DeepSeek's AI model release has had a significant impact on the stock market, with stock prices rising due to increased investor confidence and anticipation of the model's capabilities. The societal implications are vast, including potential job displacement in certain industries, increased automation, and ethical considerations surrounding AI development and deployment.",
    "summary_reasoning": "Planning to search for recent news about DeepSeek's AI model release and its market impact. Will focus on stock price movements and broader societal implications.",
    "function_calls": [
        {{
            "name": "search_google",
            "arguments": {{
                "query": "What was the impact on stock prices due to DeepSeek's AI model release, and what are the societal implications?",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31"
            }}
        }}
    ]
}}

Current Date: {date}
Current Time: {time}

### Answer with Citation
For the answer field, please cite sources with the most authoritative and reputable sources being prioritized. Use clickable links in markdown format, including the date and title of the article.

Example:
- [2024-01-01] [Title of the Article](https://www.example.com/article)

However, do not simply list up the sources with bullet points, but try to explain first in a logical narrative, and then list up the sources. It's always great to use the [INDEX] to refer to the sources and provide a detailed explanation for the sources after the narrative at the end of the answer.

### Answer with Detailed Reasoning and Source-Linked Explanations
When finalizing your answer in the "answer" field, please provide detailed reasoning and explanations with linked sources. This should enable users to fully understand the answer by reading only your final response. The process involves rationally piecing together information from sources and intermediate answers, logically connecting the dots to form a comprehensive answer.
"""

    # Format the template with our values
    return template.format(
        functions='\n'.join(function_descriptions),
        date=current_date,
        time=current_time
    )

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    MAX_ITERATIONS = 6
    # Generate system prompt using the new function
    system_prompt = make_system_prompt(tools_metadata)
    safe_log(system_prompt, "SYSTEM_PROMPT")
    output_schema = build_dynamic_output_schema(tools_metadata)

    agent = Agent(
        system_prompt=system_prompt,
        query="What was the impact on stock prices due to DeepSeek's AI model release, and what are the societal implications?",
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