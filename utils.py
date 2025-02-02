from pydantic import BaseModel, Field, create_model
from typing import Any, Union, List, Dict

def build_function_call_model(func_meta: Dict[str, Any], model_name="FunctionCall") -> type[BaseModel]:
    """
    Given a single function metadata dict:
      {
        "name": "compute_integral",
        "parameters": {
          "type": "object",
          "properties": { ... },
          "required": [ ... ],
          "additionalProperties": false
        }
      }
    returns a Pydantic model that represents:
      name: str
      arguments: ...
    with the 'arguments' field carefully built from the function's parameters.
    """

    # Step 1) The 'name' is a required string set to EXACTLY the function name
    #         If you want it to be an enum, we can do so. Or we can just store it as a string. 
    #         But for parse() to be super strict, you might do:
    from enum import Enum
    FuncNameEnum = Enum("FuncNameEnum", [(func_meta["name"], func_meta["name"])])  # single possible enum

    # Step 2) Build a dynamic "arguments" sub-model from the function's "properties"
    #         This is a bit more advanced: we iterate properties, convert them to fields, etc.
    #         We'll do a minimal version: everything is "str" or "float" depending on the metadata, 
    #         or fallback to "Any".
    properties = func_meta["parameters"].get("properties", {})
    required = func_meta["parameters"].get("required", [])
    sub_fields = {}
    for field_name, field_info in properties.items():
        field_type = field_info.get("type","string")  # "string","number","boolean","object" ...
        # map them to python types
        if field_type == "number":
            py_type = float
        elif field_type == "boolean":
            py_type = bool
        elif field_type == "object":
            py_type = dict
        elif field_type == "array":
            py_type = list
        else:
            py_type = str  # default "string"
        
        is_required = field_name in required
        default = ... if is_required else None
        sub_fields[field_name] = (py_type, Field(default=default))

    ArgumentsModel = create_model(
        "DynamicArguments",
        __base__=BaseModel,
        **sub_fields
    )

    # Step 3) The top-level function call model: must have "name" and "arguments"
    #         and also set `additionalProperties=False` if we want strict mode
    #         For parse() to honor it, we must do something like:
    base_props = {
        "name": (FuncNameEnum, Field(..., description="Function name")),
        "arguments": (ArgumentsModel, Field(..., description="Function arguments object")),
    }
    # We'll create a dynamic model class
    FunctionCallModel = create_model(
        model_name,
        __base__=BaseModel,
        **base_props
    )
    return FunctionCallModel

def build_output_model(
    tool_models: List[type[BaseModel]],
    model_name="DynamicOutputModel"
) -> type[BaseModel]:
    """
    Creates a top-level Pydantic model with:
     - keep_going: bool
     - reason_for_keep_going: str
     - final_answer: str|None
     - function_calls: list of T, where T is union of tool models
       (but .parse() doesn't handle union at top-level easily, so we might do an 'anyOf' manually)
    """
    # If you have multiple tool models, you'd do something fancy with anyOf. 
    # But let's assume you only have one for simplicity:
    OnlyModel = tool_models[0] if tool_models else BaseModel

    from pydantic import RootModel
    
    class FunctionListModel(RootModel[List[OnlyModel]]):
        pass

    # Now build the final top-level
    fields_dict = {
        "keep_going": (bool, Field(..., description="Whether to continue processing")),
        "reason_for_keep_going": (str, Field(..., description="Explanation of why processing should continue")),
        "intermediate_answer": (Union[str, None], Field(None, description="Intermediate answer to track progress so far")),
        "final_answer": (Union[str, None], Field(None, description="Final answer when processing is complete")),
        "summary_reasoning": (str, Field(..., description="Summary of the reasoning process, including: functions called and their purpose, arguments used, outputs received, how the answer was refined based on these outputs, and details of the iterative process")),
        "function_calls": (Union[FunctionListModel, None], Field(None, description="List of function calls should be provided if keep_going is true")),
    }

    OutputModel = create_model(
        model_name,
        __base__=BaseModel,
        **fields_dict
    )

    class Config:
        extra = "forbid"

    OutputModel.Config = Config
    return OutputModel

def build_dynamic_output_schema(tools_metadata: List[Dict[str, Any]]) -> type[BaseModel]:
    """
    Creates a Pydantic model for function calling based on tools metadata.
    
    Args:
        tools_metadata: List of tool/function metadata dictionaries
        
    Returns:
        A Pydantic model class for validating function calls and responses
    """
    # Build individual models for each tool
    tool_models = []
    for tool in tools_metadata:
        if tool["type"] == "function":
            model = build_function_call_model(
                tool['function'], 
                model_name=f"{tool['function']['name']}Model"
            )
            tool_models.append(model)
    
    # Build the complete output model
    return build_output_model(tool_models, model_name="DynamicOutputModel")

# ---------------------------------------------
# Example usage
# ---------------------------------------------
if __name__ == "__main__":
    tools_metadata = [
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

    DynamicOutputModel = build_dynamic_output_schema(tools_metadata)

    print(DynamicOutputModel.model_json_schema())