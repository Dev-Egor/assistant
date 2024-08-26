from typing import List
from langchain.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

import inspect
import sys
import io


@tool
def execute_code(code_string: str) -> dict | str:
    """
    Execute the passed Python code string on the terminal.
    The code string should contain valid, executable and pure python code.
    The code should also import any required python packages.

    Args:
        code_string (str): The Python code string to be executed.
    Returns:
        dict | str: A dictionary containing the output of the code or an error message if an exception occurred.
    """
    try:
        # Extracting code from Markdown code block
        code_lines = code_string.split('\n')
        code_without_markdown = '\n'.join(code_lines)

        # Create a new namespace for code execution
        exec_namespace = {}

        # Redirecting stdout to capture print output
        saved_stdout = sys.stdout
        try:
            output = io.StringIO()
            sys.stdout = output

            # Execute the code in the new namespace
            exec(code_without_markdown, exec_namespace)

            # Collect print output
            print_output = output.getvalue().strip()
        finally:
            sys.stdout = saved_stdout

        # Collect variables and function call results
        result_dict = {}
        for name, value in exec_namespace.items():
            if callable(value):
                try:
                    result_dict[name] = value()
                except TypeError:
                    # If the function requires arguments, attempt to call it with arguments from the namespace
                    arg_names = inspect.getfullargspec(value).args
                    args = {arg_name: exec_namespace.get(arg_name) for arg_name in arg_names}
                    result_dict[name] = value(**args)
            elif not name.startswith('_'):  # Exclude variables starting with '_'
                result_dict[name] = value

        # Add print output to the dictionary
        if print_output:
            result_dict['print_output'] = print_output

        return result_dict

    except Exception as e:
        error_message = f"An error occurred: {e}"
        return error_message
    

def get_openai_tools() -> List[dict]:
    functions = [
        execute_code
    ]
    tools = [convert_to_openai_tool(f) for f in functions]
    return tools