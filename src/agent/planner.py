import json
from utils.config import get_client


def create_planner_prompt(dataset_summary):
    """
    Create prompt for LLM to analyze dataset and generate a plan.
    """
    prompt = f"""
You are a data analyst agent.

Given the following dataset summary:
{dataset_summary}

Your job is to:
1. Identify the target column (the column to predict)
2. Determine if the problem is classification or regression
3. Suggest preprocessing steps
4. Suggest which models to try

Respond ONLY in JSON format like this:

{{
  "target_column": "...",
  "task_type": "classification" or "regression",
  "preprocessing": ["step1", "step2"],
  "models": ["LogisticRegression", "RandomForest"]
}}

IMPORTANT:
- Only choose models from this list: LogisticRegression, RandomForest
- Do NOT include any other models
- Do NOT include any explanation outside JSON
"""
    return prompt


def generate_plan(dataset_summary):
    """
    Calls LLM to generate analysis plan.
    """
    client = get_client()

    prompt = create_planner_prompt(dataset_summary)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a strict data science planner. Only return valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    content = response.choices[0].message.content

    try:
        plan = json.loads(content)
    except json.JSONDecodeError:
        print("Failed to parse JSON. Raw output:")
        print(content)
        raise

    return plan