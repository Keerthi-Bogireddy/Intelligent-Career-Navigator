from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import MultiLabelBinarizer

app = Flask(__name__)

# Load trained models
rf_model = joblib.load("models/rf_model.pkl")
rules = joblib.load("models/rules.pkl")

# Preprocessing function for user input
def preprocess_input(data):
    df = pd.DataFrame([data])
    mlb = MultiLabelBinarizer()
    subjects = mlb.fit_transform(df["subjects"].str.split(", "))
    subjects_df = pd.DataFrame(subjects, columns=mlb.classes_)
    skills = mlb.fit_transform(df["skills"].str.split(", "))
    skills_df = pd.DataFrame(skills, columns=mlb.classes_)
    interests = mlb.fit_transform(df["interests"].str.split(", "))
    interests_df = pd.DataFrame(interests, columns=mlb.classes_)
    df = pd.concat([df.drop(["subjects", "skills", "interests"], axis=1), subjects_df, skills_df, interests_df], axis=1)
    # Match columns with training data
    train_cols = rf_model.feature_names_in_
    for col in train_cols:
        if col not in df.columns:
            df[col] = 0
    return df[train_cols]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/questionnaire")
def questionnaire():
    return render_template("questionnaire.html")

@app.route("/results", methods=["POST"])
def results():
    data = request.form.to_dict()
    features = preprocess_input(data)
    preds = rf_model.predict_proba(features)[0]
    top_careers = pd.Series(preds, index=rf_model.classes_).sort_values(ascending=False)[:3].to_dict()
    return render_template("results.html", predictions=top_careers)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = preprocess_input(data)
    preds = rf_model.predict_proba(features)[0]
    top_careers = pd.Series(preds, index=rf_model.classes_).sort_values(ascending=False)[:3].to_dict()
    return jsonify({"predictions": top_careers})

if __name__ == "__main__":
    app.run(debug=True)