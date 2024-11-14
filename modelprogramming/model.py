import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from preprocess import preprocess_input_data
import os
import logging

# Define a path to save the trained model and encoders
MODEL_PATH = 'accident_model.pkl'
ENCODER_PATH = 'severity_encoder.pkl'
logging.basicConfig(level=logging.INFO)

# Initialize encoders
severity_encoder = LabelEncoder()

def load_data():
    df = pd.read_csv('data/accident_data.csv')
    logging.info("Data loaded successfully.")
    return df

def encode_data(df):
    df.drop(columns=['Accident date', 'Newspaper Name', 'Header', 'News title', 'AccidentDescription', 'Month-Year'], inplace=True)
    df = df.assign(**{'Car Type': df['Car Type'].str.split(' & ')}).explode('Car Type').reset_index(drop=True)

    categorical_cols = ['Identified Location', 'Season', 'Weekday', 'Accident Type', 'Car Type']
    label_encoders = {col: LabelEncoder() for col in categorical_cols}

    for col in categorical_cols:
        df[col] = label_encoders[col].fit_transform(df[col])

    df['Severity'] = severity_encoder.fit_transform(df['Severity'])
    logging.info("Data encoded successfully.")
    return df

def train_model():
    df = load_data()
    df = preprocess_input_data(df)
    df = encode_data(df)
   
    X = df.drop(columns=['Severity'])
    y = df['Severity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(severity_encoder, ENCODER_PATH)
    logging.info("Model and encoder saved successfully.")
    
    y_pred = model.predict(X_test)
    unique_classes = sorted(list(set(y_test) | set(y_pred)))
    target_names = [severity_encoder.classes_[i] for i in unique_classes if i < len(severity_encoder.classes_)]
    report = classification_report(y_test, y_pred, target_names=target_names, labels=unique_classes, zero_division=1)
    print(report)
    
    return model

def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        return train_model()
    
    model = joblib.load(MODEL_PATH)
    global severity_encoder
    severity_encoder = joblib.load(ENCODER_PATH)
    logging.info("Model and encoder loaded successfully.")
    return model

def predict_model(model, X):
    prediction = model.predict(X)
    return severity_encoder.inverse_transform(prediction)
