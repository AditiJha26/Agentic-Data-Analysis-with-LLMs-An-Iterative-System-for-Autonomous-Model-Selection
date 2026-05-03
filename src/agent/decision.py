import json
from utils.config import get_client


def create_decision_prompt(evaluation_summary, iteration):
    """
    Prompt for deciding whether to continue improving or stop.
    """
    prompt = f"""
You are a data analyst agent evaluating model performance.

Here are the current results:
{evaluation_summary}

Current iteration: {iteration}

Your job is to decide:
- Whether the model performance is good enough
- Or if further improvements should be attempted

Rules:
- If performance is reasonable and improvements are likely small → STOP
- If performance is low or can be improved → CONTINUE

Respond ONLY in JSON:

{{
  "decision": "STOP" or "CONTINUE",
  "reason": "short explanation"
}}
"""
    return prompt


def make_decision(evaluation_summary, iteration):
    """
    Calls LLM to decide next step.
    """
    client = get_client()

    prompt = create_decision_prompt(evaluation_summary, iteration)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a strict evaluator. Return only JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    content = response.choices[0].message.content

    try:
        decision = json.loads(content)
    except json.JSONDecodeError:
        print("Failed to parse decision JSON. Raw output:")
        print(content)
        raise

    return decision