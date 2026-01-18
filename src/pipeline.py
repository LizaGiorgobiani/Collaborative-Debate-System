import json
import os
from role_assignment import assign_roles
from solver import run_solver
from reviewer import run_peer_review
from judge import run_judge

PROBLEMS_FILE = "data/problems.json"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

with open(PROBLEMS_FILE, "r") as f:
    problems = json.load(f)

def run_pipeline(problem):
    problem_id = problem.get("id", "problem_1")
    question_text = problem["question"]
    print(f"\n=== Running pipeline for {problem_id} ===")

    # Role assignment
    roles = assign_roles(question_text)

    # Run solvers
    solver_outputs = [run_solver(question_text, f"solver_{i}") for i in range(1, 4)]
    with open(os.path.join(RESULTS_DIR, f"{problem_id}_raw_solutions.json"), "w") as f:
        json.dump(solver_outputs, f, indent=2)

    # Peer reviews
    peer_reviews = run_peer_review(solver_outputs)
    with open(os.path.join(RESULTS_DIR, f"{problem_id}_peer_reviews.json"), "w") as f:
        json.dump(peer_reviews, f, indent=2)

    # Refinement: send solver + reviews back to solver prompt
    refined_solutions = []
    from openai_client import call_llm_json
    from schemas import SOLVER_SCHEMA
    PROMPTS_DIR = "prompts"

    for solver in solver_outputs:
        reviews_for_solver = [r for r in peer_reviews if r["solution_id"] == solver["solver_id"]]
        combined_input = {
            "original_solution": solver,
            "reviews": reviews_for_solver
        }
        prompt_file = os.path.join(PROMPTS_DIR, "solver.txt")
        refined = call_llm_json(prompt_file, json.dumps(combined_input), role=solver["solver_id"], schema=SOLVER_SCHEMA)
        if refined is None:
            refined = solver
        refined_solutions.append(refined)

    with open(os.path.join(RESULTS_DIR, f"{problem_id}_refined_solutions.json"), "w") as f:
        json.dump(refined_solutions, f, indent=2)

    # Judge
    judge_output = run_judge(refined_solutions, peer_reviews)
    with open(os.path.join(RESULTS_DIR, f"{problem_id}_final_judgment.json"), "w") as f:
        json.dump(judge_output, f, indent=2)

    print(f"Pipeline finished for {problem_id}")
    return judge_output

def main():
    all_judgments = []
    for problem in problems:
        judgment = run_pipeline(problem)
        all_judgments.append({"problem_id": problem.get("id", "N/A"), "judgment": judgment})
    with open(os.path.join(RESULTS_DIR, "all_judgments.json"), "w") as f:
        json.dump(all_judgments, f, indent=2)

if __name__ == "__main__":
    main()
