import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

model_filename = "trained_random_forest_model.pkl"
model = joblib.load(model_filename)


accuracy = 0.97  
confusion_matrix_data = [
    [7285, 28, 0],
    [2176, 78453, 935],
    [0, 26, 4897]
]

classification_report_text = """
              precision    recall  f1-score   support

         1.0       0.77      1.00      0.87      7313
         2.0       1.00      0.96      0.98     81564
         3.0       0.84      0.99      0.91      4923

    accuracy                           0.97     93800
   macro avg       0.87      0.98      0.92     93800
weighted avg       0.97      0.97      0.97     93800
"""

metrics_per_class = """
Class 1.0: Precision = 0.77, Recall = 1.00, F1-Score = 0.87
Class 2.0: Precision = 1.00, Recall = 0.96, F1-Score = 0.98
Class 3.0: Precision = 0.84, Recall = 0.99, F1-Score = 0.91
Class macro avg: Precision = 0.87, Recall = 0.98, F1-Score = 0.92
Class weighted avg: Precision = 0.97, Recall = 0.97, F1-Score = 0.97
"""


stress_level_mapping = {
    1: "Low Stress",
    2: "Medium Stress",
    3: "High Stress"
}


def show_model_statistics():
    st.title("Model Performance")
    
    
    st.write(f"**Accuracy:** {accuracy:.2f}")
    
    
    cm = confusion_matrix_data
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 1', 'Class 2', 'Class 3'], yticklabels=['Class 1', 'Class 2', 'Class 3'])
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)
    
    
    class_labels = ['Class 1', 'Class 2', 'Class 3']
    class_counts = [407821, 36565, 24614]  
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(class_labels, class_counts, color=['#FF9999', '#66B3FF', '#99FF99'])
    ax.set_title('Class Distribution')
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Entries')
    st.pyplot(fig)

    st.write("### Classification Report:")
    st.text(classification_report_text)

    st.write("### Metrics per Class:")
    st.text(metrics_per_class)


def show_prediction_page():
    st.title("Predict Stress Level")
    st.write("Enter the input parameters:")

    ce = st.text_input("ECG Value (cecg)", value="0.0")
    em = st.text_input("EMG Value (cemg)", value="0.0")
    ed = st.text_input("EDA Value (ceda)", value="0.0")
    tp = st.text_input("Temperature Value (ctemp)", value="25.0")
    rs = st.text_input("Respiration Rate (cresp)", value="20.0")
    ax = st.text_input("Acceleration X (cax)", value="0.0")
    ay = st.text_input("Acceleration Y (cay)", value="0.0")
    az = st.text_input("Acceleration Z (caz)", value="0.0")

    try:
        ce = float(ce)
        em = float(em)
        ed = float(ed)
        tp = float(tp)
        rs = float(rs)
        ax = float(ax)
        ay = float(ay)
        az = float(az)
    except ValueError:
        st.error("Please enter valid numeric values for all parameters.")
        return

    
    if st.button("Predict Stress Level"):
        
        input_data = np.array([ce, em, ed, tp, rs, ax, ay, az]).reshape(1, -1)
        
        
        prediction = model.predict(input_data)
        
       
        stress_level = stress_level_mapping.get(prediction[0], "Unknown Stress Level")
        
        st.write(f"**Predicted Stress Level:** {stress_level} (Class {prediction[0]})")


st.sidebar.title("Stress Detection Model")
page = st.sidebar.radio("Select a Page", ['Model Performance', 'Predict'])


if page == 'Model Performance':
    show_model_statistics()
elif page == 'Predict':
    show_prediction_page()
