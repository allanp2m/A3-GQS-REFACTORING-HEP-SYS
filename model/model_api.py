from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib

app = Flask(__name__)

df = pd.read_csv("HepatitisCdata.csv")
df.drop("Unnamed: 0", axis=1, inplace=True)

for col in ['ALB', 'ALP', 'ALT', 'CHOL', 'PROT']:
    df[col].fillna(df[col].mean(), inplace=True)

le_category = LabelEncoder()
le_sex = LabelEncoder()
df['Category'] = le_category.fit_transform(df['Category'])
df['Sex'] = le_sex.fit_transform(df['Sex'])

X = df.drop("Category", axis=1)
y = df["Category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

joblib.dump((model, scaler, le_category, le_sex), "knn_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        model, scaler, le_category, le_sex = joblib.load("knn_model.pkl")
        
        input_data = pd.DataFrame([data])
        input_data["Sex"] = le_sex.transform([input_data.at[0, "Sex"]])

        input_data = input_data[['Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE',
                                'CHOL', 'CREA', 'GGT', 'PROT']]

        X_input = scaler.transform(input_data)
        prediction = model.predict(X_input)
        proba = model.predict_proba(X_input).max()

        return jsonify({
            "prediction": int(prediction[0]),
            "label": le_category.inverse_transform(prediction)[0],
            "accuracy": round(float(proba), 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000)