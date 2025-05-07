import argparse
import os
import joblib
import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_validate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters for logistic regression
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--solver", type=str, default="liblinear")
    parser.add_argument("--penalty", type=str, choices=["l1", "l2", "elasticnet", "none"], default="l2", help="Type of regularization to apply")
    parser.add_argument("--fit_intercept", type=bool, default=True)
    parser.add_argument("--nfold", type=int, default=3)

    # SageMaker specific arguments
    parser.add_argument("--train_data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation_data_dir", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))

    args = parser.parse_args()
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(f"Parsed arguments: {args}")

    data_train = pd.read_csv(f"{args.train_data_dir}/train.csv")
    X_train = data_train.drop("label", axis=1)
    y_train = data_train["label"]

    data_validation = pd.read_csv(f"{args.validation_data_dir}/validation.csv")
    X_validation = data_validation.drop("label", axis=1)
    y_validation = data_validation["label"]

    # Create and train the Logistic Regression model
    model = LogisticRegression(
        C=args.C,
        max_iter=args.max_iter,
        solver=args.solver,
        penalty=args.penalty,
        fit_intercept=args.fit_intercept
    )

    roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True)

    nfold = args.nfold
    cv_results = cross_validate(
        model, X_train, y_train,
        cv=nfold,
        scoring=roc_auc_scorer,
        return_train_score=True
    )

    train_auc = cv_results['train_score'].mean()
    validation_auc = cv_results['test_score'].mean()

    print(f"[0]#011train-auc:{train_auc:.2f}")
    print(f"[0]#011validation-auc:{validation_auc:.2f}")

    metrics_data = {
        "hyperparameters": {
            "C": args.C,
            "max_iter": args.max_iter,
            "solver": args.solver,
            "penalty": args.penalty,
            "fit_intercept": args.fit_intercept
        },
        "binary_classification_metrics": {
            "validation:auc": {"value": validation_auc},
            "train:auc": {"value": train_auc}
        }
    }

    # Save the evaluation metrics to the location specified by output_data_dir
    metrics_location = os.path.join(args.output_data_dir, "metrics.json")

    # Save the trained model to the location specified by model_dir
    model_location = os.path.join(args.model_dir, "logistic_regression_model")

    with open(metrics_location, "w") as f:
        json.dump(metrics_data, f)

    with open(model_location, "wb") as f:
        joblib.dump(model, f)
