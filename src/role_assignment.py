from openai_client import call_llm_json
import os

PROMPTS_DIR = "prompts"

def assign_roles(problem_text):
    """
    Each LLM self-assesses its best role.
    Returns dict of solver_id -> JSON role output.
    """
    roles = {}
    for i in range(1, 4):
        solver_id = f"solver_{i}"
        prompt_file = os.path.join(PROMPTS_DIR, "role_self_assessment.txt")
        output = call_llm_json(prompt_file, problem_text, role=solver_id)
        if output is None:
            output = {
                "preferred_role": "Solver",
                "confidence": 0.8,
                "reasoning": "Defaulted to Solver."
            }
        roles[solver_id] = output
    return roles
