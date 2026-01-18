import json
import os
from concurrent.futures import ThreadPoolExecutor
from openai_client import call_llm, call_llm_json


PROBLEMS_FILE = "problems.json"
PROMPTS_DIR = "prompts"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_THREADS = 3  # For parallel peer review

#schemas

SOLVER_SCHEMA = {
    "type": "object",
    "properties": {
        "solver_id": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step": {"type": "integer"},
                    "reasoning": {"type": "string"},
                    "answer": {"type": "string"}
                },
                "required": ["step", "reasoning", "answer"]
            }
        },
        "final_answer": {"type": "string"},
        "confidence": {"type": "number"}
    },
    "required": ["solver_id", "steps", "final_answer", "confidence"]
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "reviewer_id": {"type": "string"},
        "solution_id": {"type": "string"},
        "evaluation": {
            "type": "object",
            "properties": {
                "strengths": {"type": "array", "items": {"type": "string"}},
                "weaknesses": {"type": "array", "items": {"type": "string"}},
                "errors": {"type": "array"},
                "suggested_changes": {"type": "array", "items": {"type": "string"}},
                "scores": {"type": "object"}
            },
            "required": ["strengths", "weaknesses", "errors", "suggested_changes", "scores"]
        }
    },
    "required": ["reviewer_id", "solution_id", "evaluation"]
}

REFINED_SCHEMA = SOLVER_SCHEMA

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "winner": {"type": "string"},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"}
    },
    "required": ["winner", "confidence", "reasoning"]
}

#load the problems

with open(PROBLEMS_FILE, "r") as f:
    problems = json.load(f)

#role self assesment

def assign_roles(problem_text):
    roles = {}
    for i in range(1, 4):
        role_prompt = f"Role self-assessment for solver {i}:\n{problem_text}"
        output = call_llm_json(
            os.path.join(PROMPTS_DIR, "role_self_assessment.txt"),
            role_prompt,
            role=f"solver_{i}"
        )
        roles[f"solver_{i}"] = output
    return roles

#run the pipeline for one problem

def run_pipeline(problem):
    problem_id = problem.get("id", "problem_1")
    question_text = problem["question"]
    print(f"\n=== Running pipeline for {problem_id} ===")

    #role assignment
    roles = assign_roles(question_text)

    #solvers
    print("Running Solvers...")
    solver_outputs = []
    for i in range(1, 4):
        solver_id = f"solver_{i}"
        raw_output = call_llm_json(
            os.path.join(PROMPTS_DIR, "solver.txt"),
            question_text,
            role=solver_id,
            schema=SOLVER_SCHEMA
        )
        if raw_output is None:
            raw_output = {
                "solver_id": solver_id,
                "steps": [],
                "final_answer": "",
                "confidence": 0.5
            }
        raw_output["solver_id"] = solver_id
        solver_outputs.append(raw_output)

    with open(os.path.join(RESULTS_DIR, f"{problem_id}_raw_solutions.json"), "w") as f:
        json.dump(solver_outputs, f, indent=2)

    #peer reviews
    print("Generating Peer Reviews...")
    peer_reviews = []

    def review_task(reviewer, solution):
        output = call_llm_json(
            os.path.join(PROMPTS_DIR, "reviewer.txt"),
            json.dumps(solution),  # send solution JSON to reviewer
            role=reviewer["solver_id"],
            schema=REVIEW_SCHEMA
        )
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

    #parallel execution
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = []
        for reviewer in solver_outputs:
            for solution in solver_outputs:
                if solution["solver_id"] != reviewer["solver_id"]:
                    futures.append(executor.submit(review_task, reviewer, solution))
        for f in futures:
            peer_reviews.append(f.result())

    with open(os.path.join(RESULTS_DIR, f"{problem_id}_peer_reviews.json"), "w") as f:
        json.dump(peer_reviews, f, indent=2)

    #refinement
    print("Refining Solutions...")
    refined_solutions = []
    for solver in solver_outputs:
        reviews_for_solver = [r for r in peer_reviews if r["solution_id"] == solver["solver_id"]]
        combined_input = {
            "original_solution": solver,
            "reviews": reviews_for_solver
        }
        raw_refined = call_llm_json(
            os.path.join(PROMPTS_DIR, "solver.txt"),
            json.dumps(combined_input),
            role=solver["solver_id"],
            schema=REFINED_SCHEMA
        )
        if raw_refined is None:
            raw_refined = solver
        raw_refined["solver_id"] = solver["solver_id"]
        refined_solutions.append(raw_refined)

    with open(os.path.join(RESULTS_DIR, f"{problem_id}_refined_solutions.json"), "w") as f:
        json.dump(refined_solutions, f, indent=2)

    #judge
    print("Running Judge...")
    judge_input = {
        "refined_solutions": refined_solutions,
        "peer_reviews": peer_reviews
    }
    raw_judge = call_llm_json(
        os.path.join(PROMPTS_DIR, "judge.txt"),
        json.dumps(judge_input),
        role="judge",
        schema=JUDGE_SCHEMA
    )
    if raw_judge is None:
        raw_judge = {
            "winner": refined_solutions[0]["solver_id"],
            "confidence": 0.7,
            "reasoning": "Defaulted to first solver due to invalid judge output."
        }

    with open(os.path.join(RESULTS_DIR, f"{problem_id}_final_judgment.json"), "w") as f:
        json.dump(raw_judge, f, indent=2)

    print(f"Pipeline finished for {problem_id}")
    return raw_judge

#run the pipeline for all problems

def main():
    all_judgments = []
    for problem in problems:
        judgment = run_pipeline(problem)
        all_judgments.append({"problem_id": problem.get("id", "N/A"), "judgment": judgment})

    with open(os.path.join(RESULTS_DIR, "all_judgments.json"), "w") as f:
        json.dump(all_judgments, f, indent=2)

if __name__ == "__main__":
    main()
