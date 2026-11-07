import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    top_k_accuracy_score,
    cohen_kappa_score
)


def evaluate_model(
        model,
        test_ds,
        model_name,
        seed,
        duration,
        best_epoch):

    y_true = []
    y_pred = []
    y_probs = []

    for images, labels in test_ds:

        preds = model(
            images,
            training=False
        ).numpy()

        pred_labels = np.argmax(
            preds,
            axis=1
        )

        y_true.extend(
            labels.numpy()
        )

        y_pred.extend(
            pred_labels
        )

        y_probs.extend(
            preds
        )

    y_true = np.array(
        y_true
    )

    y_pred = np.array(
        y_pred
    )

    y_probs = np.array(
        y_probs
    )

    acc = accuracy_score(
        y_true,
        y_pred
    )

    balanced_acc = (
        balanced_accuracy_score(
            y_true,
            y_pred
        )
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    top2 = top_k_accuracy_score(
        y_true,
        y_probs,
        k=2
    )

    top3 = top_k_accuracy_score(
        y_true,
        y_probs,
        k=3
    )

    kappa = (
        cohen_kappa_score(
            y_true,
            y_pred
        )
    )

    report = (
        classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0
        )
    )

    return {
        "seed": seed,
        "duration": duration,
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "top2_accuracy": top2,
        "top3_accuracy": top3,
        "kappa": kappa,
        "best_epoch": best_epoch,
        "classification_report": report,
        "confusion_matrix": cm
    }