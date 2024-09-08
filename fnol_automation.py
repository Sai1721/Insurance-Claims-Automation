# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Preprocess the data
def preprocess_data(data):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data['claim_description'])
    y = data['claim_approval']
    return X, y, vectorizer

# Train the model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model and vectorizer
def save_model(model, vectorizer, model_path, vectorizer_path):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

def main():
    # File paths
    dataset_path = 'claims_data.csv'  # Update with your dataset path
    model_path = 'fnol_model.pkl'
    vectorizer_path = 'tfidf_vectorizer.pkl'

    # Load and preprocess data
    data = load_data(dataset_path)
    X, y, vectorizer = preprocess_data(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train and evaluate the model
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Save the trained model and vectorizer
    save_model(model, vectorizer, model_path, vectorizer_path)

if __name__ == "__main__":
    main()
