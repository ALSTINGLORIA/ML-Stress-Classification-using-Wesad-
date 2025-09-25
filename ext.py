import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.under_sampling import RandomUnderSampler
#from imblearn.over_sampling import SMOTE

pickle_file_path = r"file path.pkl"


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


stress_data = ch_df[ch_df['label'].isin([1, 2, 3, 4])]


ecg_30th = np.percentile(ch_df[ch_df['label'] == 2]['cecg'], 30)
ecg_60th = np.percentile(ch_df[ch_df['label'] == 2]['cecg'], 60)

emg_30th = np.percentile(ch_df[ch_df['label'] == 2]['cemg'], 30)
emg_60th = np.percentile(ch_df[ch_df['label'] == 2]['cemg'], 60)

eda_30th = np.percentile(ch_df[ch_df['label'] == 2]['ceda'], 30)
eda_60th = np.percentile(ch_df[ch_df['label'] == 2]['ceda'], 60)

temp_30th = np.percentile(ch_df[ch_df['label'] == 2]['ctemp'], 30)
temp_60th = np.percentile(ch_df[ch_df['label'] == 2]['ctemp'], 60)

resp_30th = np.percentile(ch_df[ch_df['label'] == 2]['cresp'], 30)
resp_60th = np.percentile(ch_df[ch_df['label'] == 2]['cresp'], 60)

cax_30th = np.percentile(ch_df[ch_df['label'] == 2]['cax'], 30)
cax_60th = np.percentile(ch_df[ch_df['label'] == 2]['cax'], 60)

cay_30th = np.percentile(ch_df[ch_df['label'] == 2]['cay'], 30)
cay_60th = np.percentile(ch_df[ch_df['label'] == 2]['cay'], 60)

caz_30th = np.percentile(ch_df[ch_df['label'] == 2]['caz'], 30)
caz_60th = np.percentile(ch_df[ch_df['label'] == 2]['caz'], 60)

classified_data = stress_data.copy()
classified_data['label'] = classified_data['label'].replace({1: 0, 3: 0, 4: 0})

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
            lower_percentile = percentiles[f"{feature}_30th"]
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
    'cecg_30th': ecg_30th, 'cecg_60th': ecg_60th,
    'cemg_30th': emg_30th, 'cemg_60th': emg_60th,
    'ceda_30th': eda_30th, 'ceda_60th': eda_60th,
    'ctemp_30th': temp_30th, 'ctemp_60th': temp_60th,
    'cresp_30th': resp_30th, 'cresp_60th': resp_60th,
    'cax_30th': cax_30th, 'cax_60th': cax_60th,
    'cay_30th': cay_30th, 'cay_60th': cay_60th,
    'caz_30th': caz_30th, 'caz_60th': caz_60th
}


classified_data = categorize_and_classify(classified_data, features, percentiles)


label_counts = classified_data['label'].value_counts()
print("Number of entries for each label:")
print(label_counts)

label_0_count = label_counts.get(0, 0)  
label_1_count = label_counts.get(1, 0)
label_2_count = label_counts.get(2, 0)
label_3_count = label_counts.get(3, 0)

print(f"Label 0: {label_0_count} entries")
print(f"Label 1: {label_1_count} entries")
print(f"Label 2: {label_2_count} entries")
print(f"Label 3: {label_3_count} entries")


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
# model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

