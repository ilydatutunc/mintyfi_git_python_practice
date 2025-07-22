import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'],encoding='utf-8')
data.info()
print(f"Number of missing values per column:\n{data.isnull().sum()}")

print(f"Number of duplicate records: {data.duplicated().sum()}") #yinelenen degerler
data = data.drop_duplicates() #yinelenen degerleri at
print(f"Record count after removing duplicates: {len(data)}")

print(data['label'].value_counts()) #spam ham sayisi


data.to_csv('sms_clean.csv', index=False) 

##sinif dagilimini grafik olarak da gostermek icin. pasta grafigi
labels = data['label'].value_counts().index 
sizes = data['label'].value_counts().values

plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['lightgreen', 'salmon'])
plt.title('Percentage Distribution by Message Type')
plt.axis('equal')
plt.show()
