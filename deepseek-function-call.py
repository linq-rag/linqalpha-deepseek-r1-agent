import os
import time
import json
import requests
import logging
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Type
from pydantic import BaseModel, Field

from utils import build_dynamic_openai_schema

# -----------------------------
# Environment and Logging Setup
# -----------------------------
from dotenv import load_dotenv
load_dotenv()

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {FIREWORKS_API_KEY}"
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def safe_log(message):
    try:
        logger.info(message)
    except Exception:
        pass  # Silently ignore logging errors

def generate_short_hash(input_str: str) -> str:
    """Generate a short hash (8 characters) for a given input string."""
    return hashlib.md5(input_str.encode()).hexdigest()[:8]


# -----------------------------
# Define Math Functions (Tools)
# -----------------------------
def compute_integral(expression: str, lower_bound: float, upper_bound: float) -> Union[float, str]:
    """
    Computes the definite integral of a mathematical expression (in terms of x)
    between lower_bound and upper_bound.
    """
    import sympy as sp
    x = sp.symbols('x')
    try:
        expr = sp.sympify(expression)
        result = sp.integrate(expr, (x, lower_bound, upper_bound))
        return float(result.evalf(5))
    except Exception as e:
        return f"Error computing integral: {e}"

def prove_theorem(statement: str) -> str:
    """
    Returns a high-level outline of a proof for the given theorem statement.
    (This is a stub demonstration.)
    """
    return (
        f"Proof outline for '{statement}':\n"
        "1. Identify the antiderivative of sin(x) as -cos(x).\n"
        "2. Evaluate at the bounds: -cos(pi) = 1 and -cos(0) = -1.\n"
        "3. Subtract: 1 - (-1) = 2.\n"
        "Thus, the integral equals 2."
    )

# -----------------------------
# Define Tools Metadata and Executor Mapping
# -----------------------------
tools_metadata = [
    {
        "type": "function",
        "function": {
            "name": "compute_integral",
            "description": "Computes the definite integral of a mathematical expression (in terms of x) between given bounds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression in terms of x (e.g., 'sin(x)')"
                    },
                    "lower_bound": {
                        "type": "number",
                        "description": "Lower bound of integration."
                    },
                    "upper_bound": {
                        "type": "number",
                        "description": "Upper bound of integration."
                    }
                },
                "required": ["expression", "lower_bound", "upper_bound"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "prove_theorem",
            "description": "Provides a high-level outline for proving a theorem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "statement": {
                        "type": "string",
                        "description": "The theorem statement to prove."
                    }
                },
                "required": ["statement"],
                "additionalProperties": False
            }
        }
    }
]

tools_executor = {
    "compute_integral": compute_integral,
    "prove_theorem": prove_theorem,
}

# -----------------------------
# Define the DeepSeek Chat Completion Function (Main Thinker)
# -----------------------------
def deepseek_chat_completion(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calls the DeepSeek API via the Fireworks endpoint using a POST request.
    Expects a payload with the list of messages.
    """

    messages_payload = [
        {
            "role": "user",
            "content": str(messages)
        }
    ]
    
    payload = {
        "model": "accounts/fireworks/models/deepseek-r1",
        "max_tokens": 20480,
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.6,
        "messages": messages_payload
    }

    try:
        response = requests.post(FIREWORKS_URL, headers=HEADERS, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()
    except Exception as e:
        safe_log(f"DeepSeek API error: {e}")
        return {"error": str(e)}

# -----------------------------
# Minimal DeepSeek Client Wrapper (Mimicking OpenAI's Structure)
# -----------------------------
class ChatCompletionMessage(BaseModel):
    role: str = "assistant"
    content: str = ""
    tool_calls: Optional[List[Any]] = None
    id: Optional[str] = None

class ChatCompletionChoice(BaseModel):
    message: ChatCompletionMessage

class ChatCompletionResponse(BaseModel):
    choices: List[ChatCompletionChoice]

class DeepSeekClient:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.chat = self.ChatInterface(self)

    class ChatInterface:
        def __init__(self, parent):
            self.parent = parent
            self.completions = self

        def create(self, **kwargs) -> ChatCompletionResponse:
            messages = kwargs.get("messages", [])
            safe_log(f"Sending messages to DeepSeek: {messages}")
            response_dict = deepseek_chat_completion(messages)
            safe_log(f"DeepSeek response: {response_dict}")
            try:
                return ChatCompletionResponse.model_validate(response_dict)
            except Exception as e:
                safe_log(f"Error parsing DeepSeek response: {e}")
                return ChatCompletionResponse(choices=[ChatCompletionChoice(message=ChatCompletionMessage(content=f"Error: {str(e)}"))])

# -----------------------------
# Parse Raw Output Using OpenAI (for Structured Output Parsing)
# -----------------------------
def parse_deepseek_output_with_openai(raw_output: str, openai_client: Any, model_name: str, output_schema: Type[BaseModel]) -> Any:
    """
    Uses the OpenAI parsing endpoint to extract a valid JSON object matching our schema from the raw DeepSeek output.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Here is the data to parse from the DeepSeek main agent. Please parse it into a valid JSON object that matches the provided schema as accurately as possible:"
                        f"{raw_output}\n\n"
                        "Please convert it to valid JSON strictly matching the schema provided."
                    )
                }
            ]
        }
    ]
   
    try:
        completion = openai_client.beta.chat.completions.parse(
            model=model_name,
            messages=messages,
            response_format=output_schema
        )
        # Return the parsed message directly
        return completion.choices[0].message.parsed

    except Exception as e:
        safe_log(f"Error parsing output with OpenAI: {e}")
        # Return a default structure that matches our schema
        return {
            "keep_going": False,
            "reason": f"Error parsing output: {e}",
            "intermediate_answer": raw_output,
            "final_answer": raw_output,
            "function_calls": []
        }

# -----------------------------
# Define the Agent that Uses DeepSeek as Main Thinker
# -----------------------------
class Agent:
    def __init__(self, system_prompt: str, query: str,
                 tools_metadata: List[Dict[str, Any]], tools_executor: Dict[str, Any],
                 client: DeepSeekClient, openai_client: Any, model_name: str):
        self.system_prompt = system_prompt
        self.query = query
        self.tools_metadata = tools_metadata
        self.tools_executor = tools_executor
        self.client = client
        self.openai_client = openai_client
        self.model_name = model_name
        self.max_iterations = 6
        self.output_schema = build_dynamic_openai_schema(self.tools_metadata)

    def run(self) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.query}
        ]
        
        for iteration in range(self.max_iterations):
            safe_log(f"=== Iteration {iteration} ===")
            
            # Get response from DeepSeek
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            assistant_message = response.choices[0].message.content if response.choices else ''
            safe_log(f"DeepSeek response: {assistant_message}")
            
            try:
                # Parse and validate the output
                parsed_output = parse_deepseek_output_with_openai(
                    raw_output=assistant_message,
                    openai_client=self.openai_client,
                    model_name="gpt-4o-mini",
                    output_schema=self.output_schema
                )
                
                validated_output = self.output_schema.model_validate(parsed_output)
                safe_log(f"Validated output: {validated_output}")
                
                # Add assistant's intermediate/final answer to messages
                messages.append({
                    "role": "assistant",
                    "content": validated_output.intermediate_answer or validated_output.final_answer
                })
                
                # Execute any function calls and add results to messages
                if validated_output.function_calls:
                    tool_calls_results = []
                    # Create a set of unique function calls based on name and arguments
                    seen_calls = set()
                    unique_calls = []
                    for tool_call in validated_output.function_calls.root:
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
                            safe_log(f"Calling function '{func_name}' with args: {tool_call.arguments}")
                            tool_func = self.tools_executor[func_name]
                            args_dict = {k: v for k, v in tool_call.arguments.dict().items()}  # Convert to plain dict
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
                        "role": "function",
                        "content": json.dumps([call["result"] for call in tool_calls_results]),
                        "tool_calls": tool_calls_results
                    })
                
                # If keep_going is False, return the final answer
                if not validated_output.keep_going:
                    safe_log(f"Stopping iterations: {validated_output.reason}")
                    return validated_output.final_answer or assistant_message
                
                # Otherwise continue to next iteration with updated messages
                safe_log("Continuing to next iteration...")
                    
            except Exception as e:
                safe_log(f"Error processing output: {e}")
                return assistant_message
        
        # If we hit max iterations, return the last response
        safe_log(f"Hit maximum iterations ({self.max_iterations})")
        return assistant_message

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
You are a rational function-calling agent using DeepSeek (via Fireworks API) as your main thinker.
Your task is to solve problems by carefully considering available functions and explaining your reasoning.

### Available Functions:
{functions}

### Function Usage Guidelines:
1. Analyze the user's question carefully to determine if any functions would help provide a more accurate or complete answer.
2. When relevant, use compute_integral for numerical verification of mathematical claims.
3. Use prove_theorem for generating proof outlines of mathematical statements.
4. You can and should call multiple functions in parallel when it makes sense - for example:
   - Call both compute_integral and prove_theorem simultaneously to verify and explain a theorem
   - Make multiple compute_integral calls at once to compare different intervals
5. Always explain your reasoning for using (or not using) available functions.

### Output Schema Requirements:
Your response must be a valid JSON object with the following structure:
{{
    "keep_going": "boolean",  /* true if more function calls needed, false if complete */
    "reason": "string",      /* explanation of why more calls are/aren't needed */
    "intermediate_answer": "string | null",  /* partial results or intermediate findings */
    "final_answer": "string | null",  /* final response if complete, null if not */
    "function_calls": [    /* optional array of function calls */
        {{
            "name": "string",  /* name of the function to call */
            "arguments": {{   /* object containing function arguments */
                /* specific arguments depending on the function */
            }}
        }}
    ]
}}

Example Output:
{{
    "keep_going": true,
    "reason": "Need to verify the integral calculation numerically",
    "intermediate_answer": "Let's first calculate the integral numerically to verify our approach",
    "final_answer": null,
    "function_calls": [
        {{
            "name": "compute_integral",
            "arguments": {{
                "expression": "sin(x)",
                "lower_bound": 0,
                "upper_bound": 3.14159
            }}
        }}
    ]
}}

Current Date: {date}
Current Time: {time}
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
    # Instantiate our DeepSeek client wrapper.
    deepseek_client = DeepSeekClient(api_key=FIREWORKS_API_KEY, model="accounts/fireworks/models/deepseek-r1")
    # Instantiate OpenAI client solely for output parsing.
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    # Generate system prompt using the new function
    system_prompt = make_system_prompt(tools_metadata)
    safe_log(system_prompt)

    agent = Agent(
        system_prompt=system_prompt,
        query="Prove that the integral of sin(x) from 0 to π equals 2. Explain your reasoning step by step.",
        tools_metadata=tools_metadata,
        tools_executor=tools_executor,
        client=deepseek_client,
        openai_client=openai_client,
        model_name="accounts/fireworks/models/deepseek-r1"
    )
    
    final_answer = agent.run()
    print("Final Answer:", final_answer)
