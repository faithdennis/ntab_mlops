import argparse
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


# Function to clean the text
def clean_text(text):
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    return text


# Function to binarize labels
def binarize(label):
    return label.lower() == 'spam'


# Main function
def main(args):
    # Set local path prefix in the processing container
    local_dir = "/opt/ml/processing"

    input_data_path_spam = os.path.join(f"{local_dir}/spam-mail", "spam.csv")

    logger.info("Reading spam data from {}".format(input_data_path_spam))
    df_spam = pd.read_csv(input_data_path_spam, encoding='ISO-8859-1')
    df_spam = df_spam[['v1', 'v2']]
    df_spam.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)
    df_spam['text'] = df_spam['text'].apply(clean_text)
    df_spam['label'] = df_spam['label'].apply(binarize)

    # Split the data into features and target variable
    val_ratio = args.validation_ratio
    test_ratio = args.test_ratio

    X = df_spam['text']
    y = df_spam['label']

    # Vectorize the text using TF-IDF
    print("Vectorizing text data with TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # limit to 5K feats
    X_tfidf = tfidf_vectorizer.fit_transform(X)

    # Split data into training and validation sets
    logger.debug("Splitting data into train, validation, and test sets")

    X_train_val, X_test, y_train_val, y_test = train_test_split(X_tfidf, y, test_size=test_ratio, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_ratio, random_state=42)

    y_train = pd.DataFrame(y_train, columns=['label']) 
    X_train = pd.DataFrame(X_train)
    
    y_val = pd.DataFrame(y_val, columns=['label'])
    X_val = pd.DataFrame(X_val)
    
    y_test = pd.DataFrame(y_test, columns=['label'])
    X_test = pd.DataFrame(X_test)
    
    y_df = pd.DataFrame(y, columns=['label'])
    X_df = pd.DataFrame(X)

    train_df = pd.concat([y_train, X_train], axis=1)
    val_df = pd.concat([y_val, X_val], axis=1)
    test_df = pd.concat([y_test, X_test], axis=1)
    dataset_df = pd.concat([y, X], axis=1)

    logger.info("Train data shape after preprocessing: {}".format(train_df.shape))
    logger.info("Validation data shape after preprocessing: {}".format(val_df.shape))
    logger.info("Test data shape after preprocessing: {}".format(test_df.shape))

    # Save processed datasets to the local paths in the processing container.
    # SageMaker will upload the contents of these paths to S3 bucket
    logger.debug("Writing processed datasets to container local path.")
    train_output_path = os.path.join(f"{local_dir}/train", "train.csv")
    validation_output_path = os.path.join(f"{local_dir}/val", "validation.csv")
    test_output_path = os.path.join(f"{local_dir}/test", "test.csv")
    full_processed_output_path = os.path.join(f"{local_dir}/processed", "dataset.csv")

    logger.info("Saving train data to {}".format(train_output_path))
    train_df.to_csv(train_output_path, index=False)

    logger.info("Saving validation data to {}".format(validation_output_path))
    val_df.to_csv(validation_output_path, index=False)

    logger.info("Saving test data to {}".format(test_output_path))
    test_df.to_csv(test_output_path, index=False)

    logger.info("Saving full processed data to {}".format(full_processed_output_path))
    
    dataset_df.to_csv(full_processed_output_path, index=False)


# Parse command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args, _ = parser.parse_known_args()
    logger.info("Received arguments {}".format(args))

    # Call main func for preprocessing logic
    main(args)