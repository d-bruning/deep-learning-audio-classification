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

try:
    import wandb
except ImportError:
    wandb = None


def evaluate_model(
    model,
    test_ds,
    seed,
    duration,
    best_epoch,
    num_classes,
    class_names=None,
    log_to_wandb=False
):
    """
    Evaluate a trained model.

    Returns:
        Dictionary containing metrics,
        confusion matrix,
        predictions,
        and classification report.
    """

    y_true = []
    y_pred = []
    y_probs = []

    # ======================================
    # Collect Predictions
    # ======================================

    for images, labels in test_ds:

        probs = model(
            images,
            training=False
        ).numpy()

        preds = np.argmax(
            probs,
            axis=1
        )

        y_true.extend(
            labels.numpy()
        )

        y_pred.extend(
            preds
        )

        y_probs.extend(
            probs
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

    # ======================================
    # Metrics
    # ======================================

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    balanced_accuracy = (
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

    kappa = cohen_kappa_score(
        y_true,
        y_pred
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    # ======================================
    # Top-k Validation
    # ======================================

    max_k = min(
        3,
        num_classes
    )

    top2_accuracy = np.nan
    top3_accuracy = np.nan

    if num_classes >= 2:

        top2_accuracy = (
            top_k_accuracy_score(
                y_true,
                y_probs,
                k=2
            )
        )

    if num_classes >= 3:

        top3_accuracy = (
            top_k_accuracy_score(
                y_true,
                y_probs,
                k=3
            )
        )

    # ======================================
    # Results Dictionary
    # ======================================

    results = {

        "seed":
            seed,

        "duration":
            duration,

        "samples_evaluated":
            len(y_true),

        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "top2_accuracy":
            top2_accuracy,

        "top3_accuracy":
            top3_accuracy,

        "kappa":
            kappa,

        "best_epoch":
            best_epoch,

        "classification_report":
            report,

        "confusion_matrix":
            cm,

        "y_true":
            y_true,

        "y_pred":
            y_pred,

        "y_probs":
            y_probs
    }

    # ======================================
    # WandB Logging
    # ======================================

    if (
        log_to_wandb
        and wandb is not None
        and wandb.run is not None
    ):

        wandb.log({

            "test_accuracy":
                accuracy,

            "test_balanced_accuracy":
                balanced_accuracy,

            "test_precision":
                precision,

            "test_recall":
                recall,

            "test_f1":
                f1,

            "test_top2_accuracy":
                top2_accuracy,

            "test_top3_accuracy":
                top3_accuracy,

            "test_kappa":
                kappa,

            "best_epoch":
                best_epoch
        })

        if class_names is not None:

            wandb.log({

                "confusion_matrix":
                    wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=y_true,
                        preds=y_pred,
                        class_names=class_names
                    )
            })

    return results
