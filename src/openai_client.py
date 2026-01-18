import os
import openai
import time
import json
from datetime import datetime

#configuration

DEFAULT_MODEL = "gpt-4"
DEFAULT_TEMPERATURE = 0.3
MAX_RETRIES = 3
LOG_RESPONSES = True  #set to false to disable logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

#set api key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")


#helper functions

def log_response(role, prompt_file, input_text, output_text):
    """Save prompts and outputs for debugging."""
    if not LOG_RESPONSES:
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(LOG_DIR, f"{role}_{timestamp}.json")
    log_data = {
        "role": role,
        "prompt_file": prompt_file,
        "input_text": input_text,
        "output_text": output_text
    }
    with open(filename, "w") as f:
        json.dump(log_data, f, indent=2)


def call_llm(prompt_file, input_text, role="generic", model=None, temperature=None, retries=MAX_RETRIES):
    """
    Call OpenAI LLM with a system prompt and user input.
    Returns the text response.
    """
    model = model or DEFAULT_MODEL
    temperature = temperature or DEFAULT_TEMPERATURE

    #read system prompt
    with open(prompt_file, "r") as f:
        system_prompt = f.read()

    #retry loop
    for attempt in range(1, retries + 1):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ],
                temperature=temperature
            )
            output_text = response['choices'][0]['message']['content']

            # Optional logging
            log_response(role, prompt_file, input_text, output_text)

            return output_text

        except Exception as e:
            print(f"[Attempt {attempt}] LLM call failed: {e}")
            time.sleep(1)

    raise RuntimeError(f"LLM call failed after {retries} attempts.")


def call_llm_json(prompt_file, input_text, role="generic", schema=None, model=None, temperature=None,
                  retries=MAX_RETRIES):
    """
    Call LLM and return parsed JSON. Validate against optional schema.
    If parsing fails, returns raw text in a dictionary.
    """
    raw_output = call_llm(prompt_file, input_text, role, model, temperature, retries)

    try:
        data = json.loads(raw_output)
        if schema:
            from jsonschema import validate, ValidationError
            validate(instance=data, schema=schema)
        return data
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"[Warning] JSON parsing or validation failed: {e}")
        return {"raw_text": raw_output}



if __name__ == "__main__":
    #test
    test_output = call_llm("prompts/solver.txt", "Solve 2+2", role="solver")
    print(test_output)

    test_json = call_llm_json("prompts/solver.txt", "Solve 2+2 in structured JSON", role="solver")
    print(test_json)
