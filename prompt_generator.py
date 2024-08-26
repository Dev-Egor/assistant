from pydantic import BaseModel
import yaml
import json
import os
import functions


class FunctionCall(BaseModel):
    arguments: dict
    name: str


class PromptSchema(BaseModel):
    Information: str
    Objective: str


def generate(persona_dir):
    prompt_path = os.path.join(persona_dir,"system_prompt.yml")
    output_path =  os.path.join(persona_dir,"history.json")

    tools = functions.get_openai_tools()

    with open(prompt_path, 'r') as file:
        yaml_content = yaml.safe_load(file)
    
    prompt_schema = PromptSchema(
        Information=yaml_content.get('Information', ''),
        Objective=yaml_content.get('Objective', '')
    )

    variables = {
        "functions": tools
    }

    formatted_prompt = ""
    for _, value in prompt_schema.model_dump().items():
        formatted_value = value.format(**variables)
        formatted_value = formatted_value.replace("\n", " ")
        formatted_prompt += f"{formatted_value}"

    prompt = [
            {'role': 'system', 'content': formatted_prompt}
        ]

    with open(output_path, "w") as json_file:
        json.dump(prompt, json_file)
    print("Successfuly generated history.json file.")