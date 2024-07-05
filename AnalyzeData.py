import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping

# Load the dataset
data = pd.read_csv('/home/georgikatevski/AI/WeatherCast/weather_classification_data.csv')

print(data.head())  # Display the first few rows of the dataframe

print(data.info())
print(data.describe())
print("-----------")
print(data.isna().sum())
print("-----------")
print(f"The dataset contains {data.shape[0]} rows and {data.shape[1]} columns.")
temp_type = []

low_temp = 0  # < 15
middle_temp = 0 # 15 - 35
high_temp = 0 # > 35

for i in data['Temperature']:
  if i < 15:
    low_temp += 1
    temp_type.append('Low')
  elif i >= 15 and i < 35:
    middle_temp += 1
    temp_type.append('Middle')
  else:
    high_temp += 1
    temp_type.append('High')

data['Temperature Type'] = temp_type
print(f'high: {high_temp}, middle: {middle_temp}, low: {low_temp}')


plt.figure(figsize = (10, 6))
labels = ['< 15', '15 - 35', '> 35']
values = [low_temp, middle_temp, high_temp]
colors = ['blue', 'green', 'red']
plt.pie(values, labels = labels, autopct = '%1.2f%%', colors = colors)

plt.savefig('/home/georgikatevski/AI/WeatherCast/an1.png')
plt.close()

plt.figure(figsize = (10, 6))
sns.barplot(data = data, x = 'UV Index', y = 'Temperature', hue = 'Season')
plt.savefig('/home/georgikatevski/AI/WeatherCast/an2.png')
plt.close()

winter = data['Season'].value_counts()['Winter']
spring = data['Season'].value_counts()['Spring']
autumn = data['Season'].value_counts()['Autumn']
summer = data['Season'].value_counts()['Summer']
labels = ['Winter', 'Spring', 'Autumn', 'Summer']
values = [winter, spring, autumn, summer]
colors = ['#3515EA', '#1ADC28', '#796626', '#EEF26E']

plt.figure(figsize = (10, 6))
plt.pie(values, labels = labels, colors = colors, autopct = '%1.2f%%')
plt.savefig('/home/georgikatevski/AI/WeatherCast/an3.png')
plt.close()

plt.figure(figsize = (10, 6))
sns.boxplot(data = data, x = 'Season', y = 'Visibility (km)', color = '#12C878')
plt.savefig('/home/georgikatevski/AI/WeatherCast/an4.png')
plt.close()