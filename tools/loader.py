# tool_loader.py
import json
import importlib

def load_tools_from_json(json_file_path):
    """
    Reads JSON specifying an executor_map (which modules to import)
    and a list of function metadata objects.
    
    Returns (tools_metadata_list, tools_executor_dict)
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # The "metadata" field is your array of OpenAI function definitions
    tools_metadata = data.get("metadata", [])

    # The "executor_map" associates a function name -> module path + func name
    executor_map = data.get("executor_map", {})

    tools_executor = {}

    # For each function name in the executor map, dynamically import
    for func_name, info in executor_map.items():
        module_path = info["module"]
        function_name = info["function_name"]

        module = importlib.import_module(module_path)
        func_obj = getattr(module, function_name)
        
        # e.g., tools_executor["search_google"] = reference to search_google()
        tools_executor[func_name] = func_obj

    return tools_metadata, tools_executor
