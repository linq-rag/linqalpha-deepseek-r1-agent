import re
import time
from typing import Any, Dict, List
from pydantic import BaseModel

from openai import OpenAI
from config import safe_log, FIREWORKS_API_KEY

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
