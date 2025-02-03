import json
from typing import Any, Dict, List
from pydantic import BaseModel
from config import safe_log
from deepseek_client import get_deepseek_response
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
                                safe_log(f"Result from '{func_name}': {json.dumps(tool_result, indent=2)} with args: {json.dumps(args_dict, indent=2)}", "TOOL CALL RESULT")
                                
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