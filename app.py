from flask import Flask, render_template, request
import joblib
import pandas as pd

# Initialize app
app = Flask(__name__)

# Load model and columns
model = joblib.load("churn_model.pkl")
columns = joblib.load("columns.pkl")

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form
        tenure = float(request.form['tenure'])
        monthly = float(request.form['monthly'])
        total = float(request.form['total'])

        # Create empty dataframe with all columns
        input_data = pd.DataFrame(columns=columns)
        input_data.loc[0] = 0

        # Fill required values
        input_data['tenure'] = tenure
        input_data['MonthlyCharges'] = monthly
        input_data['TotalCharges'] = total

        # Predict
        prob = model.predict_proba(input_data)[0][1]


        if prob > 0.3:
            result = f"⚠ Customer likely to churn (Probability: {prob:.2f})"
        else:
            result = f"✅ Customer likely to stay (Probability: {prob:.2f})"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

# Run app
if __name__ == "__main__":
    app.run(debug=True)