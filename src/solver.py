from openai_client import call_llm_json
import os

from schemas import SOLVER_SCHEMA  

PROMPTS_DIR = "prompts"

def run_solver(problem_text, solver_id):
    """
    Generates structured step-by-step solution for a given solver.
    """
    prompt_file = os.path.join(PROMPTS_DIR, "solver.txt")
    solution = call_llm_json(prompt_file, problem_text, role=solver_id, schema=SOLVER_SCHEMA)
    if solution is None:
        solution = {
            "solver_id": solver_id,
            "steps": [],
            "final_answer": "",
            "confidence": 0.5
        }
    solution["solver_id"] = solver_id
    return solution
