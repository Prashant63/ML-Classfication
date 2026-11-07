# Machine Learning Assignment 2 - Activity Classification Lab

This project implements, compares, and deploys five machine-learning classifiers for human-activity recognition. It is prepared for the BITS Pilani WILP M.Tech (AIML) Machine Learning Assignment 2.

## 1. Problem statement

The objective is to classify a smartphone user's activity from engineered accelerometer and gyroscope measurements. The project trains five classification models on one common dataset, evaluates them fairly on untouched test data, and provides a Streamlit interface where a user can upload labelled test data and inspect the saved-model results.

## 2. Dataset description

The project uses the [UCI Human Activity Recognition Using Smartphones Data Set](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).

- **Task:** six-class classification: `LAYING`, `SITTING`, `STANDING`, `WALKING`, `WALKING_DOWNSTAIRS`, and `WALKING_UPSTAIRS`.
- **Instances:** 10,299 observations (7,352 official training rows and 2,947 official test rows).
- **Features:** 561 time- and frequency-domain features computed from smartphone accelerometer and gyroscope signals.
- **Target column in this repository:** `activity`.
- **Split policy:** the original UCI train/test partition is retained. No row from the supplied test set is used to fit a model.

The dataset exceeds the assignment requirement of at least 12 features and 500 instances. The original UCI feature names include a few duplicates, so the generated CSV adds the original feature number as a prefix (for example, `feature_001_...`) to make every upload column name unique.

## 3. GitHub repository link

[https://github.com/Prashant63/ML-Classfication](https://github.com/Prashant63/ML-Classfication)

## 4. Repository structure

```text
ML-Classfication/
├── app.py                         # Streamlit application
├── requirements.txt               # Deployment dependencies
├── runtime.txt                    # Streamlit Cloud Python version
├── data/
│   ├── metadata.json              # Feature and target contract
│   └── test_data.csv              # Required labelled test data
├── model/                         # Five saved fitted classifiers
├── results/                       # Metrics, reports, and confusion matrices
├── src/
│   ├── prepare_data.py            # Download and format the UCI data
│   ├── train_models.py            # Fit and save all classifiers
│   ├── evaluation_utils.py        # Shared metric and chart functions
│   └── evaluate_models.py         # Offline evaluation script
└── notebooks/
    └── Assignment_2_Walkthrough.ipynb
```

## 5. Models used and evaluation metrics

Every model was trained on the same official UCI training partition and evaluated on the same untouched official test partition. Accuracy, Precision, Recall, and F1 use predicted labels. Precision, Recall, and F1 are weighted averages across the six activity classes. AUC is macro one-vs-rest AUC, and MCC is Matthews Correlation Coefficient.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9535 | 0.9975 | 0.9552 | 0.9535 | 0.9534 | 0.9445 |
| Decision Tree | 0.8585 | 0.9211 | 0.8585 | 0.8585 | 0.8579 | 0.8302 |
| K-Nearest Neighbors | 0.8890 | 0.9821 | 0.8965 | 0.8890 | 0.8879 | 0.8683 |
| Gaussian Naive Bayes | 0.7703 | 0.9578 | 0.7947 | 0.7703 | 0.7688 | 0.7286 |
| Support Vector Machine | 0.9528 | 0.9937 | 0.9534 | 0.9528 | 0.9527 | 0.9435 |

### Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | This is the overall winner. It has the best Accuracy, AUC, F1, and MCC, showing that the scaled sensor features are highly informative for a linear multiclass boundary. The main remaining confusion is between `SITTING` and `STANDING`. |
| Decision Tree | It captures some nonlinear rules without feature scaling, but its Accuracy (0.8585) and MCC (0.8302) are substantially below the best models. The depth and minimum-leaf settings limit overfitting, but one tree still cannot represent all activity boundaries as smoothly as the stronger models. |
| K-Nearest Neighbors | Its AUC is high (0.9821), but its Accuracy and F1 are lower than Logistic Regression and SVM. Classification depends on distances to stored training rows, so it is less compact and less effective in this high-dimensional feature space. |
| Gaussian Naive Bayes | This is the lowest-scoring model on Accuracy, F1, and MCC. Its conditional-independence assumption is restrictive because many accelerometer and gyroscope features are correlated, although its AUC shows that it still ranks classes reasonably well. |
| Support Vector Machine | SVM is a close second, with Accuracy 0.9528 and F1 0.9527. It separates the activity classes very well after scaling, but Logistic Regression slightly exceeds it on every reported aggregate metric and is simpler to interpret. |
| **Overall winner for this dataset** | **Logistic Regression**, because it has the highest Accuracy (0.9535), AUC (0.9975), F1 (0.9534), and MCC (0.9445) on the untouched test set. |

## 6. Run the project locally

```bash
git clone https://github.com/Prashant63/ML-Classfication.git
cd ML-Classfication
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run once to download the public UCI archive and create the CSV files.
python src/prepare_data.py

# Train the five models and reproduce the metrics and charts.
python src/train_models.py
python src/evaluate_models.py

# Launch the interactive interface.
streamlit run app.py
```

To execute the explanatory notebook locally, install Jupyter in the same environment and open `notebooks/Assignment_2_Walkthrough.ipynb`.

## 7. Streamlit app features

- Labelled test-CSV uploader, with an included `test_data.csv` available for immediate evaluation.
- Dropdown for selecting any of the five saved classifiers.
- Six required evaluation metrics: Accuracy, AUC, Precision, Recall, F1, and MCC.
- Classification report and labelled confusion matrix.
- A comparison table for the results of all five models on the included UCI test data.
- No model retraining in the app. It only loads saved artifacts and evaluates them on labelled test data.

## 8. Streamlit Community Cloud deployment

The local app has been validated. After the repository commits are pushed, deploy it in Streamlit Community Cloud by selecting this repository's `main` branch and `app.py` entry point. The live link should be added here and to the final single-PDF submission only after deployment succeeds.

## 9. Reproducibility and academic integrity

The data-preparation, training, and evaluation scripts make the result reproducible. Read the code and notebook, run the project yourself in BITS Virtual Lab, and be prepared to explain the model choices, metric definitions, and observations in your own words. The final submission must include a genuine BITS Virtual Lab execution screenshot.
