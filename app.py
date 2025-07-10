from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
import os

app = Flask(__name__)

# Load or train the model and compute accuracy
def load_or_train_model():
    model_path = 'diabetes_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        # Load accuracy if available, otherwise set to None
        accuracy_path = 'model_accuracy.pkl'
        accuracy = None
        if os.path.exists(accuracy_path):
            with open(accuracy_path, 'rb') as file:
                accuracy = pickle.load(file)
        return model, accuracy

    # Load full dataset from CSV
    try:
        data = pd.read_csv('diabetes_data.csv')  # Load your 100,000-row dataset
    except FileNotFoundError:
        print("Error: diabetes_data.csv not found in the project directory.")
        raise

    # Preprocess data
    le_gender = LabelEncoder()
    le_smoking = LabelEncoder()
    data['gender'] = le_gender.fit_transform(data['gender'])
    data['smoking_history'] = le_smoking.fit_transform(data['smoking_history'])

    # Features and target
    X = data.drop('diabetes', axis=1)
    y = data['diabetes']

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save model
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

    # Save accuracy
    with open('model_accuracy.pkl', 'wb') as file:
        pickle.dump(accuracy, file)

    # Save encoders
    with open('le_gender.pkl', 'wb') as file:
        pickle.dump(le_gender, file)
    with open('le_smoking.pkl', 'wb') as file:
        pickle.dump(le_smoking, file)

    return model, accuracy

# Load encoders
def load_encoders():
    with open('le_gender.pkl', 'rb') as file:
        le_gender = pickle.load(file)
    with open('le_smoking.pkl', 'rb') as file:
        le_smoking = pickle.load(file)
    return le_gender, le_smoking

model, accuracy = load_or_train_model()
le_gender, le_smoking = load_encoders()

@app.route('/')
def home():
    return render_template('index.html', accuracy=accuracy)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/what-is-diabetes')
def what_is_diabetes():
    return render_template('what-is-diabetes.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        gender = le_gender.transform([data['gender']])[0]
        smoking_history = le_smoking.transform([data['smoking_history']])[0]
        features = np.array([[
            gender,
            float(data['age']),
            int(data['hypertension']),
            int(data['heart_disease']),
            smoking_history,
            float(data['bmi']),
            float(data['hba1c']),
            float(data['blood_glucose'])
        ]])
        
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'message': 'Diabetic' if prediction == 1 else 'Not Diabetic',
            'accuracy': float(accuracy) if accuracy is not None else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        data = request.get_json()
        message = data.get('message', '').lower()
        # Placeholder for backend chatbot logic; currently, client-side handles most responses
        response = "For more complex queries, I'm still learning! Try asking about diabetes symptoms or prevention."
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)