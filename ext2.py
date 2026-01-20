import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix, classification_report

pickle_file_path = r"your file path"


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


ecg_33th = np.percentile(ch_df[ch_df['label'] == 2]['cecg'], 33)
ecg_66th = np.percentile(ch_df[ch_df['label'] == 2]['cecg'], 66)

emg_33th = np.percentile(ch_df[ch_df['label'] == 2]['cemg'], 33)
emg_66th = np.percentile(ch_df[ch_df['label'] == 2]['cemg'], 66)

eda_33th = np.percentile(ch_df[ch_df['label'] == 2]['ceda'], 33)
eda_66th = np.percentile(ch_df[ch_df['label'] == 2]['ceda'], 66)

temp_33th = np.percentile(ch_df[ch_df['label'] == 2]['ctemp'], 33)
temp_66th = np.percentile(ch_df[ch_df['label'] == 2]['ctemp'], 66)

resp_33th = np.percentile(ch_df[ch_df['label'] == 2]['cresp'], 33)
resp_66th = np.percentile(ch_df[ch_df['label'] == 2]['cresp'], 66)

cax_33th = np.percentile(ch_df[ch_df['label'] == 2]['cax'], 33)
cax_66th = np.percentile(ch_df[ch_df['label'] == 2]['cax'], 66)

cay_33th = np.percentile(ch_df[ch_df['label'] == 2]['cay'], 33)
cay_66th = np.percentile(ch_df[ch_df['label'] == 2]['cay'], 66)

caz_33th = np.percentile(ch_df[ch_df['label'] == 2]['caz'], 33)
caz_66th = np.percentile(ch_df[ch_df['label'] == 2]['caz'], 66)

classified_data = stress_data.copy()
# classified_data['label'] = classified_data['label'].replace({1: 0, 3: 0, 4: 0})

def categorize(value, lower_percentile, upper_percentile):
    if value <= lower_percentile:
        return 1  
    elif value <= upper_percentile:
        return 2 
    else:
        return 3  
    
feature_weight = {
    'cay': 0.177044,
    'cax': 0.164656,
    'caz': 0.126613,
    'cecg': 0.118972,
    'cresp': 0.115870,
    'ctemp': 0.108665,
    'ceda': 0.095974,
    'cemg': 0.092805
}



def categorize_and_classify_weighted(data, features, percentiles, feature_importances):
    classified_data = data.copy()
    for index, row in classified_data[classified_data['label'] == 2].iterrows():
        temp_categories = []

        for feature in features:
            lower_percentile = percentiles[f"{feature}_33th"]
            upper_percentile = percentiles[f"{feature}_66th"]

            value = row[feature]
            category = categorize(value, lower_percentile, upper_percentile)

            # Weight the category by the feature's importance
            weighted_category = category * feature_importances[feature]
            temp_categories.append(weighted_category)

        # Compute the weighted average of the categories
        weighted_average_category = sum(temp_categories) / sum(feature_importances.values())

        # Assign a new label based on the weighted average
        new_label = 1 if weighted_average_category <= 1.5 else (2 if weighted_average_category <= 2.5 else 3)
        classified_data.at[index, 'label'] = new_label

    return classified_data

features = ['cecg', 'cemg', 'ceda', 'ctemp', 'cresp', 'cax', 'cay', 'caz']
percentiles = {
    'cecg_33th': ecg_33th, 'cecg_66th': ecg_66th,
    'cemg_33th': emg_33th, 'cemg_66th': emg_66th,
    'ceda_33th': eda_33th, 'ceda_66th': eda_66th,
    'ctemp_33th': temp_33th, 'ctemp_66th': temp_66th,
    'cresp_33th': resp_33th, 'cresp_66th': resp_66th,
    'cax_33th': cax_33th, 'cax_66th': cax_66th,
    'cay_33th': cay_33th, 'cay_66th': cay_66th,
    'caz_33th': caz_33th, 'caz_66th': caz_66th
}


classified_data = categorize_and_classify_weighted(classified_data, features, percentiles,feature_weight)

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
