import streamlit as st  
import pickle as pickle
import pandas as pd  
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def apply_custom_css():
    st.markdown(
        """
        <style>
            /* Background color for the app */
            body {
                background-color: #f5f5f5;
                font-family: 'Poppins', sans-serif;
            }
            /* Sidebar styling */
            .sidebar .sidebar-content {
                background: #fff;
                padding: 20px;
            }
            /* Title styling */
            h1 {
                color: #2c3e50;
                font-weight: 600;
                text-align: center;
                margin-bottom: 20px;
            }
            /* Prediction area styling */
            .diagnosis {
                font-size: 24px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
                text-align: center;
            }
            .diagnosis.benign {
                background-color: #27ae60;
                color: white;
            }
            .diagnosis.malignant {
                background-color: #c0392b;
                color: white;
            }
            /* Radar chart styling */
            .plotly-graph {
                margin-top: -20px;
            }
        </style>
        """, unsafe_allow_html=True)



def get_clean_data():
    data = pd.read_csv("assets/breast-cancer.csv")
    data = data.drop('id', axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})

    return data

def add_sidebar():
    st.sidebar.header("Cell cytosis Measurements")

    data = get_clean_data()

    input_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    for label, key in input_labels:
        min_value = float(data[key].min())
        max_value = float(data[key].max())
        mean_value = float(data[key].mean())

        st.sidebar.write(f"**{label}**")
        st.sidebar.write(f"Range: {min_value:.2f} - {max_value:.2f}")

        input_dict[key] = st.sidebar.number_input(
            label=f"Enter {label.lower()}",
            min_value=min_value,
            max_value=max_value,
            value=mean_value,
            step=(max_value - min_value) / 100  # Adjust step size as needed
        )

    return input_dict

def get_scaled_values(input_dict):
    data = get_clean_data()

    X = data.drop("diagnosis", axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_value = X[key].max()
        min_value = X[key].min() 
        scaled_value = (value - min_value) / (max_value - min_value)
        scaled_dict[key] = scaled_value

    return scaled_dict


def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)

    categories = ['Radius', 'Texture', "Perimeter", 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points', 'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=True
    )

    return fig

def show_feature_importance():
    data = get_clean_data()
    X = data.drop("diagnosis", axis=1)
    y = data["diagnosis"]

    # Handling imbalance using SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Train the CatBoost model
    model = CatBoostClassifier(verbose=0, random_state=42)
    model.fit(X_train, y_train)

    # Plot feature importance
    feature_importance = model.get_feature_importance()
    feature_names = X.columns
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})

    plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance", y="Feature", data=importance_df.sort_values(by="Importance", ascending=False))
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")

    st.pyplot(plt)

def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", 'rb'))
    scaler = pickle.load(open("model/scaler.pkl", 'rb'))

    input_array = np.array(list(input_data.values())).reshape(1, -1)

    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.write("Predictions Area")

    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malignant'>Malignant</span>", unsafe_allow_html=True)

    
    st.write(f"Probability of being Benign: {model.predict_proba(input_array_scaled)[0][0] * 100:.2f}%")
    st.write(f"Probability of being Malignant: {model.predict_proba(input_array_scaled)[0][1] * 100:.2f}%")

    st.write("This app is meant to assist professionals in making diagnosis and should not be used for personal purpose. It is not made to substitute medical professionals.")


def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor App",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    input_data = add_sidebar()
    # st.write(input_data)


    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("This app will help in the Diagnosis of breast cancer")
    
    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)

    # Adding feature importance section
    st.header("Feature Importance Analysis")
    show_feature_importance()


if __name__ == "__main__":
    main()
