import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix, classification_report


pickle_file_path = r"C:/Users/alsti/Desktop/Coding Projects/Main Project AJCE- Stress Level Classification/archive/WESAD/S8/S8.pkl"


data = pd.read_pickle(pickle_file_path)
print("Pickle file loaded successfully!")


cax = data['signal']['chest']['ACC'][0:,0]
cay = data['signal']['chest']['ACC'][0:,1]
caz = data['signal']['chest']['ACC'][0:,2]
cecg = data['signal']['chest']['ECG'][:,0]
cemg = data['signal']['chest']['EMG'][:,0]
ceda = data['signal']['chest']['EDA'][:,0]
ctemp = data['signal']['chest']['Temp'][:,0]
cresp = data['signal']['chest']['Resp'][:,0]
label = data['label']



chest = [cax, cay, caz, cecg, cemg, ceda, ctemp, cresp, label] 
ch_array = np.array(chest) 
ch_array = ch_array.T 
Columns = ['cecg', 'cemg','ceda','ctemp', 'cresp','cax', 'cay', 'caz', 'label' ]
ch_df = pd.DataFrame(ch_array, columns = Columns) 


stress_data = ch_df[ch_df['label'].isin([2])]


ecg_35th = np.percentile(ch_df[ch_df['label'] == 2]['cecg'], 35)
ecg_60th = np.percentile(ch_df[ch_df['label'] == 2]['cecg'], 60)

emg_35th = np.percentile(ch_df[ch_df['label'] == 2]['cemg'], 35)
emg_60th = np.percentile(ch_df[ch_df['label'] == 2]['cemg'], 60)

eda_35th = np.percentile(ch_df[ch_df['label'] == 2]['ceda'], 35)
eda_60th = np.percentile(ch_df[ch_df['label'] == 2]['ceda'], 60)

temp_35th = np.percentile(ch_df[ch_df['label'] == 2]['ctemp'], 35)
temp_60th = np.percentile(ch_df[ch_df['label'] == 2]['ctemp'], 60)

resp_35th = np.percentile(ch_df[ch_df['label'] == 2]['cresp'], 35)
resp_60th = np.percentile(ch_df[ch_df['label'] == 2]['cresp'], 60)

cax_35th = np.percentile(ch_df[ch_df['label'] == 2]['cax'], 35)
cax_60th = np.percentile(ch_df[ch_df['label'] == 2]['cax'], 60)

cay_35th = np.percentile(ch_df[ch_df['label'] == 2]['cay'], 35)
cay_60th = np.percentile(ch_df[ch_df['label'] == 2]['cay'], 60)

caz_35th = np.percentile(ch_df[ch_df['label'] == 2]['caz'], 35)
caz_60th = np.percentile(ch_df[ch_df['label'] == 2]['caz'], 60)

classified_data = stress_data.copy()
# classified_data['label'] = classified_data['label'].replace({1: 0, 3: 0, 4: 0})

def categorize(value, lower_percentile, upper_percentile):
    if value <= lower_percentile:
        return 1  
    elif value <= upper_percentile:
        return 2 
    else:
        return 3  
    
def categorize_and_classify(data, features, percentiles):
    classified_data = data.copy()

    for index, row in classified_data[classified_data['label'] == 2].iterrows():
        temp_categories = []

        for feature in features:
            lower_percentile = percentiles[f"{feature}_35th"]
            upper_percentile = percentiles[f"{feature}_60th"]

            value = row[feature]

            category = categorize(value, lower_percentile, upper_percentile)
            temp_categories.append(category)

        average_category = sum(temp_categories) / len(temp_categories)

        new_label = 1 if average_category <= 1.5 else (2 if average_category <= 2.5 else 3)

        classified_data.at[index, 'label'] = new_label

    return classified_data

features = ['cecg', 'cemg', 'ceda', 'ctemp', 'cresp', 'cax', 'cay', 'caz']
percentiles = {
    'cecg_35th': ecg_35th, 'cecg_60th': ecg_60th,
    'cemg_35th': emg_35th, 'cemg_60th': emg_60th,
    'ceda_35th': eda_35th, 'ceda_60th': eda_60th,
    'ctemp_35th': temp_35th, 'ctemp_60th': temp_60th,
    'cresp_35th': resp_35th, 'cresp_60th': resp_60th,
    'cax_35th': cax_35th, 'cax_60th': cax_60th,
    'cay_35th': cay_35th, 'cay_60th': cay_60th,
    'caz_35th': caz_35th, 'caz_60th': caz_60th
}


classified_data = categorize_and_classify(classified_data, features, percentiles)


label_counts = classified_data['label'].value_counts()
print("Number of entries for each label:")
print(label_counts)



X = classified_data[['cecg', 'cemg', 'ceda', 'ctemp', 'cresp', 'cax', 'cay', 'caz']] 
y = classified_data['label'] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
"""
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
"""
undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
model = RandomForestClassifier(n_estimators=20, random_state=42,class_weight='balanced')
model.fit(X_train_resampled, y_train_resampled)


y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


report = classification_report(y_test, y_pred, output_dict=True)

print("\nMetrics per Class:")
for label in report:
    if label != 'accuracy':
        print(f"Class {label}: Precision = {report[label]['precision']:.2f}, "
              f"Recall = {report[label]['recall']:.2f}, "
              f"F1-Score = {report[label]['f1-score']:.2f}")

model_filename = "trained_random_forest_model.pkl"
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")