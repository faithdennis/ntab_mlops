import json
import logging
import pathlib
import pickle
import tarfile
import pandas as pd
from sklearn.metrics import roc_auc_score

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    logger.debug("Loading logistic regression model.")
    model = pickle.load(open("logistic_regression_model", "rb"))

    logger.debug("Reading test data.")
    test_local_path = "/opt/ml/processing/test/test.csv"
    df_test = pd.read_csv(test_local_path)

    # Extract test set target column
    y_test = df_test.iloc[:, 0].values

    cols_when_train = model.feature_names
    # Extract test set feature columns
    X_test = df_test[cols_when_train].copy()

    logger.info("Generating predictions for test data.")
    pred_probs = model.predict_proba(X_test)[:, 1]

    # Calculate model evaluation score
    logger.debug("Calculating ROC-AUC score.")
    auc = roc_auc_score(y_test, pred_probs)
    metric_dict = {
        "classification_metrics": {"roc_auc": {"value": auc}}
    }

    # Save model evaluation metrics
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing evaluation report with ROC-AUC: %f", auc)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(metric_dict))