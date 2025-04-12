# Intelligent Career Navigator

A mini-project to predict career options using user survey data and ML.
### To Create A virtual environment
    1. cmd in this folder
    2. Type the following :->  python -m venv venv
    3. To activate the environment type the following :-> venv\Scripts\activate
## Setup
1. Install dependencies:
   
   pip install flask pandas scikit-learn mlxtend joblib

2. Prepare data:
    Place survey_data.csv in data/ folder.
3. Train models:
    python train_model.py
4. Run the app:
    python app.py
Open http://127.0.0.1:5000 in your browser.
## Features
    Home page with career options.
    Questionnaire to collect user data.
    Results page with top 3 career predictions.
## Future Enhancements
    Add association rule suggestions.
    Integrate live job trends via APIs.
## How to Run the Project
1. **Prepare Environment**:
   - Install Python 3.x and required libraries (`pip install flask pandas scikit-learn mlxtend joblib`).
   - Create the folder structure as above.
   - Convert your Excel data to `survey_data.csv` and place it in `data/`.

2. **Train Models**:
   - Run `python train_model.py` to preprocess data and save models to `models/`.

3. **Start Server**:
   - Run `python app.py` and visit `http://127.0.0.1:5000` in your browser.

4. **Test**:
   - Navigate from Home → Questionnaire → Results.
   - Input sample data (e.g., Subjects: "Math, Physics", Skills: "Coding, Problem-solving").

---

## Notes
- **Simplifications**: The questionnaire is trimmed for brevity (add more fields from your survey as needed). The preprocessing assumes comma-separated inputs for multi-select fields.
- **Expansion**: Add more career pathways from your list, enhance the UI, or integrate live updates using APIs (e.g., Indeed for job trends).
- **Data**: Ensure your `survey_data.csv` matches the expected columns (adjust `preprocess.py` if needed).