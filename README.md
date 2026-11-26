# Hotel Booking Cancellation Predictor

CDS 4003 – Data Science for Business Application (Stage 7: Deployment)

A Streamlit web app that predicts whether a hotel booking will be cancelled, explains the main drivers,
and estimates the revenue at risk, so reservation and revenue teams can decide what to do.

**Live app:** https://cds4003-hotel-cancellation.streamlit.app/

**GitHub:** https://github.com/amnasaif300/hotel-booking-cancellation-prediction

## Team
| Name | Student ID |
|---|---|
| Sara Yousif | H00535688 |
| Amna Saif | H00495503 |

## What the app does
| Tab | Purpose |
|---|---|
| Predict a booking | Enter one booking, get a cancellation probability, risk level, suggested action, revenue at risk, and a "why" chart |
| Batch scoring | Upload a CSV of bookings, get a ranked list with risk levels, download the result |
| Model performance | Metrics, threshold trade-off, feature importance, calibration, model comparison, limitations |
| About | Business problem, method, data source |

The sidebar lets the hotel change the decision threshold (default **0.35**, chosen in Stage 6) and the
assumed chance that a cancelled room is re-sold.

## Project stages
| Stage | File |
|---|---|
| 3 Data cleaning | `notebooks/01_Hotel_Booking_Data_Preparation.ipynb` |
| 4 EDA and features | `notebooks/02_Hotel_Booking_EDA_Feature_Engineering.ipynb` |
| 5 Model selection | `notebooks/03_Hotel_Booking_Model_Selection.ipynb` |
| 6 Tuning and evaluation | `notebooks/04_Hotel_Booking_Hyperparameter_Tuning_Evaluation.ipynb` |
| 7 Deployment | `app/streamlit_app.py` |

## Final model
Gradient Boosting (`HistGradientBoostingClassifier`), tuned with RandomizedSearchCV:
`learning_rate=0.05, max_iter=200, max_leaf_nodes=31, min_samples_leaf=30`.
Threshold 0.35 (best F1 on cross-validated training predictions).

Test set (17,195 bookings): ROC-AUC 0.914. At threshold 0.35: recall 79.9%, precision 67.6%, F1 0.732, accuracy 83.9%.
At the default 0.50: recall 66.2%, precision 75.1%, F1 0.704.

## Repository structure
```
app/            streamlit_app.py (the app), model_utils.py (prediction and explanation logic)
src/            train_model.py (re-creates the model from the Stage 4 data)
models/         hotel_cancellation_model.joblib, model_metadata.json
data/           hotel_bookings.csv, hotel_bookings_cleaned.csv, hotel_bookings_model_ready.csv
notebooks/      numbered notebooks for Stages 3-6
docs/           model comparison tables from Stages 5 and 6
tests/          unit tests for the app logic
requirements.txt
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```
Re-create the model (optional, takes about a minute):
```bash
python src/train_model.py --data data/hotel_bookings_model_ready.csv
```
Run the tests: `python -m pytest tests`

## Deploy on Streamlit Community Cloud
1. Push this folder to a **public** GitHub repository (see below).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Repository: your repo. Branch: `main`. Main file path: `app/streamlit_app.py`.
5. Open **Advanced settings** and choose **Python 3.12** (or newer). The pinned scikit-learn needs Python 3.11+.
6. Click **Deploy**. The first build takes a few minutes.
7. The live URL is listed above.

The deployed GitHub repository uses a flat layout with `streamlit_app.py`, `model_utils.py` and both model files at its root. The archive retains the original organized layout for local use. The deployed model paths were adjusted to match the flat layout.

If the app shows "The model could not be loaded", first check the model file paths and then check whether the scikit-learn version differs from the one the model was saved with. Re-run `python src/train_model.py` in an environment with the versions in `requirements.txt`,
commit the new files in `models/`, and push.

### Uploading to GitHub without the command line
Create a new public repository, click **Add file → Upload files**, drag in the *contents* of this folder
(keep the folder structure), and commit. Files up to 25 MB can be uploaded in the browser; all files here are smaller.

## Data
Hotel booking demand dataset (City Hotel and Resort Hotel, Portugal, 2015-2017): Antonio, N., de Almeida, A.
and Nunes, L. (2019), *Hotel booking demand datasets*, Data in Brief, 22, 41-49.
Source: [Kaggle – Hotel Booking Demand (Jesse Mostipak)](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand). Licence: CC BY 4.0.

## Limitations
- Trained on two Portuguese hotels, 2015-2017; behaviour and policies may have changed. The app uses 2017 as the arrival year.
- Duplicate rows were removed for modelling (training cancellation rate 27.6% vs 37.1% in the full data), so real-world precision and recall may differ.
- Country and agent are strong predictors but may reflect these hotels' own channels.
- Predictions support staff decisions and should not trigger automatic overbooking.

