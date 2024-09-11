import pandas as pd  
import numpy as np  
from catboost import CatBoostClassifier 
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay, accuracy_score, confusion_matrix, precision_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE



def get_data():
    data = pd.read_csv(r"C:\Users\UK-PC\Desktop\sample-project-2\BREAST CANCER CLASSIFICATION-4\assets\breast-cancer.csv")
    data = data.drop('id', axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})
    return data

def create_model(data):
    X = data.drop("diagnosis", axis=1)
    y = data['diagnosis']

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = CatBoostClassifier()
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    print(f"accuracy of the model: {accuracy_score(y_test, predictions)}")
    print(f"Classification Report: {classification_report(y_test, predictions)}")
    print(f"Precision Score: {precision_score(y_test, predictions)}")

    return model, scaler



def main():
    data = get_data()

    model, scaler = create_model(data)

    with open(r"C:\Users\UK-PC\Desktop\sample-project-2\BREAST CANCER CLASSIFICATION-4\model\model.pkl", 'wb') as f:
        pickle.dump(model, f)  
    
    with open(r"C:\Users\UK-PC\Desktop\sample-project-2\BREAST CANCER CLASSIFICATION-4\model\scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)


if __name__ == "__main__":
    main()
