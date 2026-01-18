from concurrent.futures import ThreadPoolExecutor
from openai_client import call_llm_json
import os
from schemas import REVIEW_SCHEMA

PROMPTS_DIR = "prompts"
MAX_THREADS = 3

def run_peer_review(solver_outputs):
    """
    Generates peer reviews in parallel for all solver solutions.
    Returns list of review JSONs.
    """
    reviews = []

    def review_task(reviewer, solution):
        prompt_file = os.path.join(PROMPTS_DIR, "reviewer.txt")
        output = call_llm_json(prompt_file, json.dumps(solution), role=reviewer["solver_id"], schema=REVIEW_SCHEMA)
        if output is None:
            output = {
                "reviewer_id": reviewer["solver_id"],
                "solution_id": solution["solver_id"],
                "evaluation": {
                    "strengths": [],
                    "weaknesses": [],
                    "errors": [],
                    "suggested_changes": [],
                    "scores": {"correctness": 0.5, "clarity": 0.5}
                }
            }
        return output

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = []
        for reviewer in solver_outputs:
            for solution in solver_outputs:
                if solution["solver_id"] != reviewer["solver_id"]:
                    futures.append(executor.submit(review_task, reviewer, solution))
        for f in futures:
            reviews.append(f.result())

    return reviews
