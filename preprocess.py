import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np

def preprocess_data(file_path, output_path="processed_dataset.csv"):
    # Load the data
    df = pd.read_csv(file_path)
    print("Original columns:", df.columns.tolist())
    print("Original shape:", df.shape)

    # Step 1: Clean the data
    df = df.drop_duplicates(subset="id", keep="first")
    df.columns = [col.strip().replace(" ", "_").replace("\n", "_") for col in df.columns]

    # Drop the 'extracurricular_activities' column if it exists
    if 'extracurricular_activities' in df.columns:
        df = df.drop(columns=['extracurricular_activities'])

    # Handle missing values (fill with median for numerical, mode for categorical)
    numerical_cols = ['cgpa', 'math_score', 'physics_score', 'biology_score', 'history_score',
                      'weekly_self_study_hours', 'budget','career_demand_score', 'openness',
                      'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    categorical_cols = ['education_level', 'interests', 'career_aspiration', 'hobbies']

    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Fix age
    df['age'] = pd.to_numeric(df['age'], errors="coerce").fillna(df['age'].median())

    # Encode categorical variables
    le = LabelEncoder()
    for col in ['education_level', 'interests', 'career_aspiration']:
        df[col] = le.fit_transform(df[col].astype(str))
    print("Encoded classes for career_aspiration:", le.classes_)

    # Handle hobbies (one-hot encoding)
    hobby_cols = ['coding', 'cooking', 'dance', 'finance', 'gaming', 'history', 'music',
                  'painting', 'reading', 'research', 'science', 'sports', 'travel', 'writing']
    for hobby in hobby_cols:
        df[hobby] = df['hobbies'].str.contains(hobby, case=False, na=False).astype(int)
    df = df.drop('hobbies', axis=1)

    # Scale numerical features
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Prepare X and y
    X = df.drop(['id', 'career_aspiration'], axis=1)
    y = df['career_aspiration']

    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to '{output_path}'")
    print("Processed shape:", df.shape)

    return X, y, le.classes_

if __name__ == "__main__":
    file_path = r"C:\Users\Akshitha Kotte\Desktop\mini-project-new\Intelligent-Career-Navigator\data\icn_dataset.csv"
    try:
        X, y, career_classes = preprocess_data(file_path)
        print("\nSample X (first 5 rows):")
        print(X.head())
        print("\nSample y (first 5 rows):")
        print(y.head())
    except Exception as e:
        print(f"An error occurred: {e}")