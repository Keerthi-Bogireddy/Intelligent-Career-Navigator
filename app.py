from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('best_model.pkl')

# Initialize encoders and scaler (these should match the training process)
le_edu = LabelEncoder()
le_interests = LabelEncoder()
le_career = LabelEncoder()  # Assuming career_aspiration classes are stored in the model training
scaler = MinMaxScaler()

# Dummy fit for encoders (update with actual training data classes if available)
le_edu.fit(['Grade 10', 'Grade 12', 'UG', 'PG'])
le_interests.fit(['arts', 'business', 'healthcare', 'technology'])
le_career.fit(['Arts', 'CA', 'Data Scientist', 'Defence Persenal', 'Engineering', 'Finance Manager',
               'Historian', 'IT Specialist', 'Lawyer', 'MBA', 'Medicine', 'Software Engineer'])

# Feature columns (match with training data)
numerical_cols = ['age', 'cgpa', 'math_score', 'physics_score', 'biology_score', 'history_score',
                  'weekly_self_study_hours', 'career_demand_score', 'openness',
                  'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
hobby_cols = ['coding', 'cooking', 'dance', 'finance', 'gaming', 'history', 'music', 'painting',
              'reading', 'research', 'science', 'sports', 'travel', 'writing']  # Added 'cooking'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    data = {
        'age': float(request.form['age']),
        'education_level': le_edu.transform([request.form['education_level']])[0],
        'cgpa': float(request.form['cgpa']),
        'math_score': float(request.form['math_score']),
        'physics_score': float(request.form['physics_score']),
        'biology_score': float(request.form['biology_score']),
        'history_score': float(request.form['history_score']),
        'openness': float(request.form['openness']),
        'conscientiousness': float(request.form['conscientiousness']),
        'extraversion': float(request.form['extraversion']),
        'agreeableness': float(request.form['agreeableness']),
        'neuroticism': float(request.form['neuroticism']),
        'weekly_self_study_hours': float(request.form['weekly_self_study_hours']),
        'extracurricular_activities': int(request.form['extracurricular_activities']),
        'interests': le_interests.transform([request.form['interests']])[0],
        'budget': float(request.form['budget']),
        'career_demand_score': float(request.form['career_demand_score'])
    }

    # Handle hobbies (checkboxes)
    hobbies = request.form.getlist('hobbies')
    for hobby in hobby_cols:
        data[hobby] = 1 if hobby in hobbies else 0

    # Convert to DataFrame
    input_df = pd.DataFrame([data])

    # Ensure all feature names match training (add missing columns with default value 0)
    expected_features = numerical_cols + hobby_cols
    for feature in expected_features:
        if feature not in input_df.columns:
            input_df[feature] = 0

    # Reorder columns to match training data
    input_df = input_df[expected_features]

    # Scale numerical features
    input_df[numerical_cols] = scaler.fit_transform(input_df[numerical_cols])

    # Make prediction
    prediction = model.predict(input_df)
    predicted_career = le_career.inverse_transform(prediction)[0]  # Convert encoded prediction back to label

    return render_template('result.html', career=predicted_career)

if __name__ == '__main__':
    app.run(debug=True)