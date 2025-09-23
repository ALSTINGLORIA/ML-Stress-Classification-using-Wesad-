import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

pickle_file_path = r"your file name here.pkl"


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


ecg_25th = np.percentile(ch_df[ch_df['label'] == 2]['cecg'], 25)
ecg_75th = np.percentile(ch_df[ch_df['label'] == 2]['cecg'], 75)

emg_25th = np.percentile(ch_df[ch_df['label'] == 2]['cemg'], 25)
emg_75th = np.percentile(ch_df[ch_df['label'] == 2]['cemg'], 75)

eda_25th = np.percentile(ch_df[ch_df['label'] == 2]['ceda'], 25)
eda_75th = np.percentile(ch_df[ch_df['label'] == 2]['ceda'], 75)

temp_25th = np.percentile(ch_df[ch_df['label'] == 2]['ctemp'], 25)
temp_75th = np.percentile(ch_df[ch_df['label'] == 2]['ctemp'], 75)

resp_25th = np.percentile(ch_df[ch_df['label'] == 2]['cresp'], 25)
resp_75th = np.percentile(ch_df[ch_df['label'] == 2]['cresp'], 75)

cax_25th = np.percentile(ch_df[ch_df['label'] == 2]['cax'], 25)
cax_75th = np.percentile(ch_df[ch_df['label'] == 2]['cax'], 75)

cay_25th = np.percentile(ch_df[ch_df['label'] == 2]['cay'], 25)
cay_75th = np.percentile(ch_df[ch_df['label'] == 2]['cay'], 75)

caz_25th = np.percentile(ch_df[ch_df['label'] == 2]['caz'], 25)
caz_75th = np.percentile(ch_df[ch_df['label'] == 2]['caz'], 75)

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
            lower_percentile = percentiles[f"{feature}_25th"]
            upper_percentile = percentiles[f"{feature}_75th"]

            value = row[feature]

            category = categorize(value, lower_percentile, upper_percentile)
            temp_categories.append(category)

        average_category = sum(temp_categories) / len(temp_categories)

        new_label = 1 if average_category <= 1.5 else (2 if average_category <= 2.5 else 3)

        classified_data.at[index, 'label'] = new_label

    return classified_data

features = ['cecg', 'cemg', 'ceda', 'ctemp', 'cresp', 'cax', 'cay', 'caz']
percentiles = {
    'cecg_25th': ecg_25th, 'cecg_75th': ecg_75th,
    'cemg_25th': emg_25th, 'cemg_75th': emg_75th,
    'ceda_25th': eda_25th, 'ceda_75th': eda_75th,
    'ctemp_25th': temp_25th, 'ctemp_75th': temp_75th,
    'cresp_25th': resp_25th, 'cresp_75th': resp_75th,
    'cax_25th': cax_25th, 'cax_75th': cax_75th,
    'cay_25th': cay_25th, 'cay_75th': cay_75th,
    'caz_25th': caz_25th, 'caz_75th': caz_75th
}


classified_data = categorize_and_classify(classified_data, features, percentiles)



X = classified_data[['cecg', 'cemg', 'ceda', 'ctemp', 'cresp', 'cax', 'cay', 'caz']] 
y = classified_data['label'] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

