import json
from utils.config import get_client


def create_refinement_prompt(dataset_summary, current_plan, evaluation_summary):
    """
    Prompt to refine preprocessing plan.
    """
    prompt = f"""
You are a data analyst improving a machine learning pipeline.

Dataset summary:
{dataset_summary}

Current plan:
{current_plan}

Evaluation results:
{evaluation_summary}

Your job:
- Improve preprocessing steps to get better performance
- Suggest small, meaningful improvements

Respond ONLY in JSON:

{{
  "preprocessing": ["updated steps"]
}}

Rules:
- Keep changes realistic
- Do not add completely new models
- Focus on improving preprocessing
"""
    return prompt


def refine_plan(dataset_summary, current_plan, evaluation_summary):
    """
    Calls LLM to refine preprocessing.
    """
    client = get_client()

    prompt = create_refinement_prompt(
        dataset_summary,
        current_plan,
        evaluation_summary
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    content = response.choices[0].message.content

    try:
        refined = json.loads(content)
    except json.JSONDecodeError:
        print("Refinement JSON failed:")
        print(content)
        raise

    return refined