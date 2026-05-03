def format_results(results):
    """
    Convert model results into a readable string for the LLM.
    """
    output = "Model Performance Summary:\n"

    for model_name, metrics in results.items():
        acc = metrics["accuracy"]
        f1 = metrics["f1_score"]

        output += f"{model_name}:\n"
        output += f"  Accuracy: {acc:.4f}\n"
        output += f"  F1 Score: {f1:.4f}\n\n"

    return output


def get_best_model(results):
    """
    Identify the best model based on F1 score.
    """
    best_model = None
    best_score = -1

    for model_name, metrics in results.items():
        if metrics["f1_score"] > best_score:
            best_score = metrics["f1_score"]
            best_model = model_name

    return best_model, best_score


def summarize_evaluation(results):
    """
    Combine formatted results and best model selection.
    """
    formatted = format_results(results)
    best_model, best_score = get_best_model(results)

    summary = formatted
    summary += f"Best Model: {best_model} with F1 Score = {best_score:.4f}\n"

    return summary