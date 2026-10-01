from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate_classification_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm
    }


def summarize_model_evaluation(evaluation_results):
    report = evaluation_results["classification_report"]

    return {
        "accuracy": evaluation_results["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"]
    }


if __name__ == "__main__":
    y_true = ["A", "A", "B", "B"]
    y_pred = ["A", "B", "B", "B"]

    results = evaluate_classification_model(
        y_true,
        y_pred
    )
    print(summarize_model_evaluation(results))
