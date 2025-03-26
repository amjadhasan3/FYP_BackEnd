from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import shap

app = Flask(__name__)
CORS(app) # allows cross-origin requests for Flutter


def preprocess_data(df):
    df = df.copy()

    # log transformation
    if "gross" in df.columns:
        df["log_gross"] = np.log1p(df["gross"])
    df["log_budget"] = np.log1p(df["budget"])

    # feature engineering
    df["budget_score_ratio"] = df["log_budget"] / (df["score"] + 1)
    df["budget_vote_ratio"] = df["budget"] / (df["votes"] + 1)
    df["budget_runtime_ratio"] = df["budget"] / (df["runtime"] + 1)
    df["vote_year_ratio"] = df["votes"] / (df["year"] - df["year"].min() + 1)
    df["vote_score_ratio"] = df["votes"] / (df["score"] + 1)
    df["budget_year_ratio"] = df["log_budget"] / (df["year"] - df["year"].min() + 1)
    df["votes_per_year"] = df["votes"] / (df["year"] - df["year"].min() + 1)
    df["score_runtime_ratio"] = df["score"] / (df["runtime"] + 1)
    df["budget_per_minute"] = df["budget"] / (df["runtime"] + 1)
    df["is_high_score"] = (df["score"] >= df["score"].quantile(0.75)).astype(int)
    df["is_high_budget"] = (df["log_budget"] >= df["log_budget"].quantile(0.75)).astype(int)
    df["is_high_votes"] = (df["votes"] >= df["votes"].quantile(0.75)).astype(int)
    df["is_recent"] = (df["year"] >= df["year"].quantile(0.75)).astype(int)

    categorical_features = ["released", "writer", "rating", "name", "genre", "director", "star", "country", "company"]

    # encodes categorical features into numerical values using LabelEncoder
    for feature in categorical_features:
        df[feature] = df[feature].astype(str)
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])

    numerical_features = ["runtime", "score", "year", "votes", "log_budget", "budget_vote_ratio", "budget_runtime_ratio",
                          "budget_score_ratio", "vote_score_ratio", "budget_year_ratio", "vote_year_ratio",
                          "score_runtime_ratio", "budget_per_minute", "votes_per_year", "is_recent", "is_high_budget",
                          "is_high_votes", "is_high_score"]

    # handles missing values
    imputer = SimpleImputer(strategy="median")
    df[numerical_features] = imputer.fit_transform(df[numerical_features])

    # standardize numerical features
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    if "gross" in df.columns:
        df = df.drop(["gross", "budget"], axis=1)
    else:
        df = df.drop(["budget"], axis=1)

    return df


def prepare_features(df):
    processed_df = preprocess_data(df)

    if "log_gross" in processed_df.columns:
        y = processed_df["log_gross"]
        X = processed_df.drop("log_gross", axis=1)
    else:
        y = None
        X = processed_df

    return X, y

def run_stacked_model():
    df = pd.read_csv("movie_dataset.csv")
    X, y = prepare_features(df)

    # base models
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, min_samples_split=10, min_samples_leaf=5)
    dt_model = DecisionTreeRegressor(max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42)
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, max_depth=5)

    # train base models
    rf_model.fit(X, y)
    dt_model.fit(X, y)
    xgb_model.fit(X, y)

    # meta model input
    meta_X_train = pd.DataFrame({
        'rf': rf_model.predict(X),
        'dt': dt_model.predict(X),
        'xgb': xgb_model.predict(X)
    })

    # train meta model using RandomForestRegressor
    meta_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, min_samples_split=10,
                                       min_samples_leaf=5)
    meta_model.fit(meta_X_train, y)

    return rf_model, dt_model, xgb_model, meta_model, X.columns

rf_model, dt_model, xgb_model, meta_model, feature_names = run_stacked_model()


def predict_gross_stacked(input_data, rf_model, dt_model, xgb_model, meta_model, feature_names):
    processed_data = preprocess_data(pd.DataFrame([input_data]))
    for feature in feature_names:
        if feature not in processed_data.columns:
            processed_data[feature] = 0
    processed_data = processed_data[feature_names]

    rf_prediction = rf_model.predict(processed_data)
    dt_prediction = dt_model.predict(processed_data)
    xgb_prediction = xgb_model.predict(processed_data)

    meta_input = pd.DataFrame({
        'rf': rf_prediction,
        'dt': dt_prediction,
        'xgb': xgb_prediction
    })
    log_prediction = meta_model.predict(meta_input)
    prediction = np.exp(log_prediction) - 1
    return prediction[0]

def explain_prediction(input_data, rf_model, dt_model, xgb_model, meta_model, feature_names):
    processed_data = preprocess_data(pd.DataFrame([input_data]))

    for feature in feature_names:
        if feature not in processed_data.columns:
            processed_data[feature] = 0
    processed_data = processed_data[feature_names]

    rf_explainer = shap.TreeExplainer(rf_model)
    rf_shap_values = rf_explainer.shap_values(processed_data)

    dt_explainer = shap.TreeExplainer(dt_model)
    dt_shap_values = dt_explainer.shap_values(processed_data)

    xgb_explainer = shap.TreeExplainer(xgb_model)
    xgb_shap_values = xgb_explainer.shap_values(processed_data)


    meta_input = pd.DataFrame({
        'rf': rf_model.predict(processed_data),
        'dt': dt_model.predict(processed_data),
        'xgb': xgb_model.predict(processed_data)
    })
    meta_explainer = shap.TreeExplainer(meta_model)
    meta_shap_values = meta_explainer.shap_values(meta_input)

    # store SHAP values in a dictionary
    explanations = {
        "rf": {},
        "dt": {},
        "xgb": {},
        "meta": {}
    }
    # populate the dictionary with SHAP values for each base model
    for i, feature in enumerate(feature_names):
      # convert to standard float to ensure JSON serializability
       explanations["rf"][feature] = float(rf_shap_values[0][i])
       explanations["dt"][feature] = float(dt_shap_values[0][i])
       explanations["xgb"][feature] = float(xgb_shap_values[0][i])

    meta_features = ['rf', 'dt', 'xgb']
    for i, feature in enumerate(meta_features):
         explanations["meta"][feature] = float(meta_shap_values[0][i])

    return explanations


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # get the JSON data from the request body
        data = request.get_json()
        # generate the prediction by passing the input data and models
        prediction = predict_gross_stacked(data, rf_model, dt_model, xgb_model, meta_model, feature_names)
        # return the prediction as a JSON response by converting the python dictionary to a JSON object
        return jsonify({'prediction': prediction})
    except Exception as e:
        # return an error message as a JSON response with a 400 status code Bad Request
        return jsonify({'error': str(e)}), 400


@app.route('/explain', methods=['POST'])
def explain():
    try:
        # get the JSON data from the request body
        data = request.get_json()
        # generate the explanation by passing the input data and models
        explanation = explain_prediction(data, rf_model, dt_model, xgb_model, meta_model, feature_names)
        # return the prediction as a JSON response by converting the python dictionary to a JSON object
        return jsonify({'explanation': explanation})
    except Exception as e:
        # return an error message as a JSON response with a 400 status code Bad Request
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
