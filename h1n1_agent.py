# h1n1_agent.py

import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Paths where the trained artifacts will be saved
MODEL_PATH = "h1n1_model.joblib"
ENCODER_PATH = "h1n1_encoder.joblib"

def _prepare_features(
    df: pd.DataFrame,
    encoder: OneHotEncoder = None,
):
    """
    - Automatically detect categorical vs numeric columns.
    - Fill missing values:
        • categoricals → 'Unknown'
        • numerics      → 0
    - If `encoder` is None, fit a new OneHotEncoder on all categorical columns.
      Otherwise, reuse the provided encoder (for prediction).
    - Return (X_full_df, fitted_encoder, y_series_or_None).
    """

    # 1) Drop respondent_id if present (never a feature)
    if "respondent_id" in df.columns:
        df = df.drop(columns=["respondent_id"])

    # 2) Separate out the y (if present)
    y = None
    if "h1n1_vaccine" in df.columns:
        y = df["h1n1_vaccine"].astype(int)
        df = df.drop(columns=["h1n1_vaccine"])

    # 3) If encoder is None, figure out which columns are categorical vs numeric.
    #    Otherwise, force categorical_cols to be exactly what encoder was fitted on.
    if encoder is None:
        # a) Identify by dtype
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols     = df.select_dtypes(include=["int64", "float64", "Int64", "Float64"]).columns.tolist()
    else:
        # b) Reuse exactly the categorical columns that the encoder saw during fit
        #    (OneHotEncoder stores them in feature_names_in_)
        categorical_cols = list(encoder.feature_names_in_)
        #    Any numeric columns are "everything else" that is not in categorical_cols
        numeric_cols     = [c for c in df.columns if c not in categorical_cols]

    # 4) Fill missing values
    #    For categorical: ensure we have every column in `categorical_cols`. If a column is
    #    missing from df, create it with all values = "Unknown".
    df_cat = df.reindex(columns=categorical_cols)      # this will put NaN in any missing columns
    df_cat = df_cat.fillna("Unknown")

    #    For numeric: keep only the numeric_cols (and fill missing with 0). If a numeric_col
    #    was absent from df, `df.reindex(columns=numeric_cols)` will create it with NaN.
    df_num = df.reindex(columns=numeric_cols).fillna(0)

    # 5) One‑hot encode the categorical columns
    if encoder is None:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_cat_enc = enc.fit_transform(df_cat)
    else:
        enc = encoder
        X_cat_enc = enc.transform(df_cat)   # now df_cat has exactly the right columns

    #    Build a DataFrame from one‑hot encodings (use the encoder's output names)
    cat_feature_names = enc.get_feature_names_out(categorical_cols)
    df_cat_encoded = pd.DataFrame(
        X_cat_enc,
        columns=cat_feature_names,
        index=df.index,
    )

    # 6) Combine numeric + one‑hot categorical into a single DataFrame
    df_num = df_num.reset_index(drop=True)
    df_cat_encoded = df_cat_encoded.reset_index(drop=True)

    X_full = pd.concat([df_cat_encoded, df_num], axis=1)
    return X_full, enc, y


def train_h1n1_model(data_folder: str = "data"):
    """
    Expects:
      data_folder/training_set_features.csv
      data_folder/training_set_labels.csv

    Merges on `respondent_id`, then:
      • Automatically splits off "h1n1_vaccine" as y.
      • Detects categoricals vs numerics, fits OneHotEncoder.
      • Trains a LogisticRegression.
      • Saves:
          - h1n1_model.joblib
          - h1n1_encoder.joblib

    Returns a dict:
      {
        "message": "...",
        "train_accuracy": 0.xxx,
        "test_accuracy": 0.yyy,
        "model_path": "...",
        "encoder_path": "..."
      }
    or {"error": "..."} if files are missing.
    """
    # Build paths
    features_path = os.path.join(data_folder, "training_set_features.csv")
    labels_path   = os.path.join(data_folder, "training_set_labels.csv")

    # Check existence
    if not os.path.exists(features_path):
        return {"error": f"Training features not found at {features_path}"}
    if not os.path.exists(labels_path):
        return {"error": f"Training labels not found at {labels_path}"}

    # Load CSVs
    df_feat = pd.read_csv(features_path)
    df_lab  = pd.read_csv(labels_path)

    # Ensure respondent_id is present
    if "respondent_id" not in df_feat.columns or "respondent_id" not in df_lab.columns:
        return {"error": "`respondent_id` column missing in features or labels CSV."}

    # Merge on respondent_id
    df = pd.merge(df_feat, df_lab, on="respondent_id", how="inner")

    # Ensure h1n1_vaccine column exists after merge
    if "h1n1_vaccine" not in df.columns:
        return {"error": "`h1n1_vaccine` column missing after merge."}

    # Prepare features & encoder
    X_full, fitted_encoder, y_series = _prepare_features(df, encoder=None)
    
    feature_columns = X_full.columns.tolist()
    joblib.dump(feature_columns, "h1n1_feature_columns.joblib")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_series, test_size=0.2, random_state=42
    )

    # Train logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc  = model.score(X_test, y_test)

    # Serialize model + encoder
    joblib.dump(model, MODEL_PATH)
    joblib.dump(fitted_encoder, ENCODER_PATH)

    return {
        "message": "Model trained successfully",
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "model_path": MODEL_PATH,
        "encoder_path": ENCODER_PATH,
    }


def predict_h1n1_adoption(user_input: dict):
    """
    Expects a dict with the same feature columns (except respondent_id, h1n1_vaccine) 
    as in training_set_features.csv. E.g.:

      {
        "age_group": "25-34",
        "sex": "Male",
        "race": "White",
        "education": "College Graduate",
        "income_poverty": "At or Above Poverty",
        "employment_status": "Employed",
        "marital_status": "Married",
        "rent_or_own": "Own",
        "some_numeric_col": 2,
        ...
      }

    Returns:
      {
        "prediction": "Likely to adopt" / "Unlikely to adopt",
        "probability": 0.xxxx
      }
    or {"error": "..."} if the model/encoder are missing.
    """
    print("SG: predict_h1n1_adoption ", user_input)
    # Check for saved artifacts
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        return {"error": "Model is not trained yet. Run `train_h1n1_model` first."}

    # Load them
    model = joblib.load(MODEL_PATH)
    enc   = joblib.load(ENCODER_PATH)
    feature_columns = joblib.load("h1n1_feature_columns.joblib")

    # Build a 1-row DataFrame from user_input
    df_input = pd.DataFrame([user_input])

    # Prepare features (reuse existing encoder)
    X_full_raw, _, _ = _prepare_features(df_input, encoder=enc)
    X_full = X_full_raw.reindex(columns=feature_columns, fill_value=0)

    # Predict probability of h1n1 = 1
    proba = model.predict_proba(X_full)[0][1]
    label = int(model.predict(X_full)[0])

    return {
        "prediction": "Likely to adopt" if label == 1 else "Unlikely to adopt",
        "probability": round(float(proba), 4),
    }

def main_pred():
    return predict_h1n1_adoption(user_input={
        "h1n1_concern": 1,
        "h1n1_knowledge": 1,
        "behavioral_face_mask": 0,
        "behavioral_large_gatherings": 0,
        "behavioral_wash_hands": 0,
        "behavioral_outside_home": 0,
        "behavioral_touch_face": 0,
        "child_under_6_months": 0,
        "health_worker": 0,
        "opinion_h1n1_vacc_effective": 0,
        "opinion_h1n1_risk": 0,
        "opinion_h1n1_sick_from_vacc": 0,
        "age_group": "18 - 34 Years",
        "education": "12 Years",
        "race": "White",
        "sex": "Male",
        "income_poverty": "Below Poverty",
        "marital_status": "Not Married",
        "rent_or_own": "Rent",
        "employment_status": "Employed",
        "census_msa": "Non-MSA",
        "hhs_geo_region": "xqwwgdyp",
        "behavioral_social_distance": 0,
        "employment_industry":"atmlpfrs",
        "employment_occupation":"hfxkjkmi"
    })

if __name__ == "__main__":
    # train_h1n1_model()
    print("Pred: ", main_pred())
