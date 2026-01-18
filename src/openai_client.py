import os
import openai
import json
import time
from jsonschema import validate, ValidationError

openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL = "gpt-4"
TEMPERATURE = 0.3
MAX_RETRIES = 3

def call_llm(prompt_file, input_text, model=None, temperature=None, retries=MAX_RETRIES):
    model = model or MODEL
    temperature = temperature or TEMPERATURE

    with open(prompt_file, "r") as f:
        prompt = f.read()

    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": input_text}
                ],
                temperature=temperature
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            print(f"[Attempt {attempt+1}] LLM call failed: {e}")
            time.sleep(1)
    raise RuntimeError(f"LLM call failed after {retries} attempts.")

def call_llm_json(prompt_file, input_text, role=None, schema=None):
    """
    Calls LLM and validates output against JSON schema.
    Returns structured JSON or None if invalid.
    """
    raw = call_llm(prompt_file, input_text)
    try:
        data = json.loads(raw)
        if schema:
            validate(instance=data, schema=schema)
        if role:
            data["solver_id"] = role
        return data
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"JSON validation failed for role={role}: {e}")
        return None
