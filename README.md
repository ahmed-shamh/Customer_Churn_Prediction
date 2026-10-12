# Telco Customer Churn Prediction

An end-to-end Machine Learning project that predicts whether a telecom
customer is likely to leave the service (**churn**). The notebook covers
data inspection, cleaning, exploratory analysis, preprocessing, model
comparison, holdout evaluation, threshold analysis, error analysis,
feature importance, and model export.

## Project Goals

-   Explore customer data and identify patterns associated with churn.
-   Build a classification pipeline that handles numerical and
    categorical features.
-   Compare a simple baseline with Logistic Regression and Random
    Forest.
-   Evaluate performance using metrics suited to an imbalanced
    classification problem.
-   Examine how changing the decision threshold affects recall and
    precision.
-   Inspect prediction errors and estimate feature importance.

## Dataset

-   **Source:** [Telco Customer Churn ---
    Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
-   **Target:** `Churn` (`Yes` / `No`)
-   **Identifier excluded from predictors:** `customerID`

Download the CSV file named `WA_Fn-UseC_-Telco-Customer-Churn.csv` from
Kaggle.

## Workflow

1.  **Load and audit data** --- inspect dimensions, data types, missing
    values, and target distribution.
2.  **Clean data** --- convert `TotalCharges` to numeric, handle
    whitespace-only values as missing, and exclude the customer
    identifier from model features.
3.  **Explore the data** --- review churn distribution and relationships
    involving tenure, monthly charges, total charges, and categorical
    attributes.
4.  **Split the data** --- create a stratified 80/20 training and
    holdout-test split.
5.  **Preprocess features** --- median imputation and standardization
    for numerical columns; most-frequent imputation and one-hot encoding
    for categorical columns.
6.  **Compare models** --- evaluate a most-frequent baseline, Logistic
    Regression, and Random Forest using 5-fold stratified
    cross-validation on the training set.
7.  **Select and evaluate** --- select the model using mean
    cross-validation ROC-AUC, then evaluate it on the holdout set.
8.  **Analyze predictions** --- review the confusion matrix, ROC curve,
    precision-recall curve, decision-threshold behavior, and prediction
    errors.
9.  **Estimate feature importance** --- use permutation importance with
    ROC-AUC scoring.
10. **Export artifacts** --- save the fitted pipeline and feature
    metadata for reuse.

## Models and Evaluation

Models included: - Dummy Classifier (`most_frequent`) --- baseline -
Logistic Regression - Random Forest

Metrics reported: - Accuracy - Precision - Recall - F1-score - ROC-AUC -
Average Precision

Because the churn classes are imbalanced, accuracy alone is not
sufficient to judge the model.

## Reported Results

The notebook's saved output reports the following holdout performance
for **Logistic Regression**:

  Metric                Score
  ------------------- -------
  Accuracy              0.738
  Precision             0.504
  Recall                0.783
  F1-score              0.614
  ROC-AUC               0.841
  Average Precision     0.633

A threshold of **0.48** is reported to achieve a recall of **0.802** in
the notebook's threshold analysis. The feature with the highest
permutation-based model reliance in the saved output is **`tenure`**.

These figures are specific to this dataset split and notebook run;
results may differ if the data, dependencies, or random seed change.

## Important Evaluation Note

The notebook uses the holdout test predictions for threshold analysis
after reporting the default-threshold test metrics. This makes the
threshold analysis a **diagnostic**, not a fully independent final test
of the selected threshold. For a production-quality evaluation, choose
the operating threshold using validation data, then evaluate that
threshold once on an untouched test set.

Feature importance describes model reliance, not causation.
Relationships observed in the dataset should not be interpreted as proof
that a feature causes churn.

## Requirements

The notebook uses Python and the following main libraries:

-   pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   scikit-learn
-   joblib

Install the dependencies with:

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## How to Run

### Option A --- Kaggle

1.  Open the notebook in Kaggle.
2.  Add the Telco Customer Churn dataset as an input.
3.  Run the notebook cells from top to bottom.

The notebook searches `/kaggle/input` for
`WA_Fn-UseC_-Telco-Customer-Churn.csv`.

### Option B --- Run locally

1.  Download the CSV from Kaggle.
2.  Place the CSV beside the notebook, or update `CSV_PATH` in the
    data-loading cell to point to its location.
3.  Install the requirements listed above.
4.  Run all notebook cells in order.

## Generated Artifacts

The notebook creates an `artifacts/` directory and saves the fitted
model pipeline and related feature metadata there. Run the notebook to
generate these files.

## Repository Contents

-   `customer-churn-prediction.ipynb` --- end-to-end notebook.
-   `README.md` --- project overview, workflow, results, and run
    instructions.
-   `WA_Fn-UseC_-Telco-Customer-Churn.csv` --- dataset downloaded
    separately from Kaggle; include it only if its license and
    repository policies permit.

## Limitations

-   This is a learning and portfolio project based on a historical
    public dataset.
-   The model's performance may not generalize to current customers or
    another telecom company.
-   Threshold selection depends on the business trade-off between missed
    churn cases and false alarms.
-   Predictions should support further analysis, not be treated as
    guaranteed outcomes.

## License and Data

Refer to the [Kaggle dataset
page](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
for dataset details and usage terms. Add a repository license if you
intend to specify how your own code may be reused.
