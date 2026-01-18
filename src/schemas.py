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
