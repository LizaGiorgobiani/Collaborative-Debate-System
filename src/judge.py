from openai_client import call_llm_json
import os
from schemas import JUDGE_SCHEMA

PROMPTS_DIR = "prompts"

def run_judge(refined_solutions, peer_reviews):
    """
    Evaluates refined solutions and selects the winner.
    """
    judge_input = {
        "refined_solutions": refined_solutions,
        "peer_reviews": peer_reviews
    }
    prompt_file = os.path.join(PROMPTS_DIR, "judge.txt")
    output = call_llm_json(prompt_file, json.dumps(judge_input), role="judge", schema=JUDGE_SCHEMA)
    if output is None:
        output = {
            "winner": refined_solutions[0]["solver_id"],
            "confidence": 0.7,
            "reasoning": "Defaulted to first solver due to invalid judge output."
        }
    return output
