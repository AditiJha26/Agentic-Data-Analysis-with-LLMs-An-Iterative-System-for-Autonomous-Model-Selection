from tools.data_tool import summarize_dataset
from tools.model_tool import train_models
from tools.eval_tool import summarize_evaluation
from agent.planner import generate_plan
from agent.decision import make_decision
from agent.refiner import refine_plan
import json


def run_agent(df, max_iterations=5):
    """
    Main agent loop.
    """
    print("Starting Agent...\n")

    # Step 1: Summarize dataset
    dataset_summary = summarize_dataset(df)

    # Step 2: Generate initial plan
    plan = generate_plan(dataset_summary)

    print("Initial Plan:")
    print(plan)
    print("\n" + "="*50 + "\n")

    target_column = plan["target_column"]

    iteration = 1

    while iteration <= max_iterations:
        print(f"Iteration {iteration}")

        # Step 3: Train models using current plan
        preprocessing_steps = plan.get("preprocessing", [])
        results = train_models(df, target_column, preprocessing_steps)

        # Step 4: Evaluate results
        evaluation_summary = summarize_evaluation(results)

        print("Evaluation:")
        print(evaluation_summary)

        # Step 5: Decision
        decision = make_decision(evaluation_summary, iteration)

        print("Decision:")
        print(decision)

        print("\n" + "-"*50 + "\n")

        # Step 6: Check stopping condition
        if decision["decision"] == "STOP" and iteration > 1:
            print("Agent decided to STOP.")
            break

        # Step 7: Refine plan (NEW)
        print("🔧 Refining plan...")

        try:
            refined = refine_plan(dataset_summary, plan, evaluation_summary)

            # Update preprocessing only
            new_steps = refined.get("preprocessing", plan["preprocessing"])

            if new_steps != plan["preprocessing"]:
                print("Updated preprocessing steps:")
                print(new_steps)
                plan["preprocessing"] = new_steps
            else:
                print("No meaningful change in preprocessing.")

        except Exception as e:
            print("Refinement failed, continuing with current plan.")
            print(e)

        print("\n" + "="*50 + "\n")

        iteration += 1

    print("Agent finished.\n")

    with open("../outputs/reports/final_output.json", "w") as f: 
        json.dump({
            "results": results,
            "decision": decision,
            "plan": plan
        }, f, indent=4)

    return {
        "final_results": results,
        "final_decision": decision,
        "final_plan": plan
    }