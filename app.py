from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

app = Flask(__name__)

# Load the trained model, scaler, and label encoders
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
le_edu = joblib.load('label_encoder_edu.pkl')
le_interests = joblib.load('label_encoder_interests.pkl')
le_career = joblib.load('label_encoder_career.pkl')

# Dummy map for extracurricular activities
extra_activities_map = {
    'none': 0,
    'sports': 1,
    'music': 2,
    'arts': 3,
    'volunteering': 4,
    'others': 5
}

# Feature columns (match with training data)
numerical_cols = ['age', 'cgpa', 'math_score', 'physics_score', 'biology_score', 'history_score',
                  'openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
hobby_cols = ['coding', 'cooking', 'dance', 'finance', 'gaming', 'history', 'music', 'painting',
              'reading', 'research', 'science', 'sports', 'travel', 'writing']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictor')
def predictor():
    return render_template('predictor.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
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
        }

        # Handle extracurricular activities
        extracurricular = request.form['extracurricular_activities']
        data['extracurricular_activities'] = extra_activities_map.get(extracurricular, 0)
        
        # Handle hobbies (convert to 1/0 based on checkbox)
        for hobby in hobby_cols:
            data[hobby] = int(request.form.get(hobby, 0))

        # Convert data to DataFrame
        df = pd.DataFrame([data])

        # Scaling the data using the saved scaler
        df_scaled = scaler.transform(df[numerical_cols])

        # Prediction
        prediction = model.predict(df_scaled)
        predicted_career = le_career.inverse_transform(prediction)

        return render_template('result.html', career=predicted_career[0])
    
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
