import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

try:
    model = pickle.load(open('titanic_model.pkl', 'rb'))
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Model file 'titanic_model.pkl' not found.")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def index():
    # Render the HTML page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text='Error: Model is not loaded.')

    try:
        # 1. Get data from form
        pclass = int(request.form['pclass'])
        sex = request.form['sex']
        age = float(request.form['age'])

        siblings_spouses = int(request.form['siblings_spouses'])
        parents_children = int(request.form['parents_children'])
        
        fare = float(request.form['fare'])

        family_size = siblings_spouses + parents_children + 1
        is_alone = 1 if family_size == 1 else 0

        # 2. Create a DataFrame from the input
        input_data = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [sex],
            'Age': [age],
            'Fare': [fare],
            'FamilySize': [family_size],
            'IsAlone': [is_alone]
        })
        
        print(f"Input data for prediction:\n{input_data}")

        # 3. Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        # 4. Format output
        if prediction[0] == 1:
            output = "This passenger would have SURVIVED."
        else:
            output = "This passenger would NOT have survived."
            
        # Get the probability
        proba_percent = f"{prediction_proba[0][prediction[0]] * 100:.2f}%"
        output_prob = f"Confidence: {proba_percent}"

    except Exception as e:
        output = f"An error occurred: {e}"
        output_prob = ""

    return render_template('index.html', prediction_text=output, prediction_prob=output_prob)

if __name__ == "__main__":
    app.run(debug=True)

