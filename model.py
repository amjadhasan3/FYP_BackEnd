import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap

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
    df = pd.read_csv("output.csv")
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # base models
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, min_samples_split=10,
                                     min_samples_leaf=5)
    dt_model = DecisionTreeRegressor(max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42)
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, max_depth=5)

    # train base models
    rf_model.fit(X_train, y_train)
    dt_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    # create predictions for meta model
    rf_predictions = rf_model.predict(X_test)
    dt_predictions = dt_model.predict(X_test)
    xgb_predictions = xgb_model.predict(X_test)

    # meta model input
    meta_X_train = pd.DataFrame({
        'rf': rf_model.predict(X_train),
        'dt': dt_model.predict(X_train),
        'xgb': xgb_model.predict(X_train)
    })
    meta_X_test = pd.DataFrame({
        'rf': rf_predictions,
        'dt': dt_predictions,
        'xgb': xgb_predictions
    })

    # train meta model using RandomForestRegressor
    meta_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, min_samples_split=10,
                                       min_samples_leaf=5)
    meta_model.fit(meta_X_train, y_train)

    # evaluate Base models
    print("Evaluating Base Models:")
    rf_mse, rf_r2, rf_mae = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    dt_mse, dt_r2, dt_mae = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    xgb_mse, xgb_r2, xgb_mae = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    # evaluate meta model
    stacked_predictions = meta_model.predict(meta_X_test)
    mse = mean_squared_error(y_test, stacked_predictions)
    r2 = r2_score(y_test, stacked_predictions)
    mae = mean_absolute_error(y_test, stacked_predictions)

    print("Meta Model Performance:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R2: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")

    return rf_model, dt_model, xgb_model, meta_model, X.columns, X_test, y_test, stacked_predictions

def evaluate_model(model, X_test, y_test, model_name="Model"):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print(f"{model_name} Performance:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R2: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    return mse, r2, mae

def predict_gross_stacked(input_data, rf_model, dt_model, xgb_model, meta_model, feature_names):
    processed_data = preprocess_data(pd.DataFrame([input_data]))

    # ensure all features are present
    for feature in feature_names:
        if feature not in processed_data.columns:
            # if a feature is missing add it with a value of 0 to handle errors
            processed_data[feature] = 0
    # select only the relevant features and ensure correct order
    processed_data = processed_data[feature_names]

    # make predictions using the base models
    rf_prediction = rf_model.predict(processed_data)
    dt_prediction = dt_model.predict(processed_data)
    xgb_prediction = xgb_model.predict(processed_data)

    # create the input for the meta model
    meta_input = pd.DataFrame({
        'rf': rf_prediction,
        'dt': dt_prediction,
        'xgb': xgb_prediction
    })
    # make the final prediction using the meta-model
    log_prediction = meta_model.predict(meta_input)
    # exponentiate the log prediction and subtract 1 to get the predicted gross revenue in the original scale
    prediction = np.exp(log_prediction) - 1
    # returns a NumPy array so we extract the first element of the gross revenue
    return prediction[0]

def explain_prediction(input_data, rf_model, dt_model, xgb_model, meta_model, feature_names):
    processed_data = preprocess_data(pd.DataFrame([input_data]))

    # ensure all features are present
    for feature in feature_names:
        if feature not in processed_data.columns:
            # if a feature is missing add it with a value of 0 to handle errors
            processed_data[feature] = 0
    # select only the relevant features and ensure correct order
    processed_data = processed_data[feature_names]

    # explain the predictions of the base models using SHAP and gives the SHAP values for each base model
    rf_explainer = shap.TreeExplainer(rf_model)
    rf_shap_values = rf_explainer.shap_values(processed_data)
    dt_explainer = shap.TreeExplainer(dt_model)
    dt_shap_values = dt_explainer.shap_values(processed_data)
    xgb_explainer = shap.TreeExplainer(xgb_model)
    xgb_shap_values = xgb_explainer.shap_values(processed_data)

    # create the input for the meta model
    meta_input = pd.DataFrame({
        'rf': rf_model.predict(processed_data),
        'dt': dt_model.predict(processed_data),
        'xgb': xgb_model.predict(processed_data)
    })
    # explain the predictions of the meta model using SHAP and gives the SHAP values
    meta_explainer = shap.TreeExplainer(meta_model)
    meta_shap_values = meta_explainer.shap_values(meta_input)

    print("SHAP values for RF Model:")
    for i, feature in enumerate(feature_names):
        print(f"{feature}: {rf_shap_values[0][i]:.4f}")

    print("\nSHAP values for DT Model:")
    for i, feature in enumerate(feature_names):
        print(f"{feature}: {dt_shap_values[0][i]:.4f}")

    print("\nSHAP values for XGB Model:")
    for i, feature in enumerate(feature_names):
        print(f"{feature}: {xgb_shap_values[0][i]:.4f}")

    print("\nSHAP values for Meta Model:")
    meta_features = ['rf', 'dt', 'xgb']
    for i, feature in enumerate(meta_features):
        print(f"{feature}: {meta_shap_values[0][i]:.4f}")

    # visualizaton for meta model
    shap.force_plot(meta_explainer.expected_value, meta_shap_values[0], meta_input.iloc[0], matplotlib=True)

if __name__ == "__main__":
    input_data = {
        "released": "December",
        "writer": "Christopher Nolan",
        "rating": "R",
        "name": "Glory",
        "genre": "Action",
        "director": "Christopher Nolan",
        "star": "Tom Cruise",
        "country": "USA",
        "company": "Universal",
        "runtime": 133,
        "score": 0,
        "budget": 120,
        "year": 2021,
        "votes": 8,
    }
    rf_model, dt_model, xgb_model, meta_model, feature_names, X_test, y_test, stacked_predictions = run_stacked_model()
    predicted_gross = predict_gross_stacked(input_data, rf_model, dt_model, xgb_model, meta_model, feature_names)
    print(f'\nPredicted Revenue for "{input_data["name"]}": ${predicted_gross:,.2f}')
    explain_prediction(input_data, rf_model, dt_model, xgb_model, meta_model, feature_names)
