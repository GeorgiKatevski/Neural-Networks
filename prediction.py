import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('/home/georgikatevski/AI/WeatherCast/weather_classification_data.csv')
print(data.head())  # Display the first few rows of the dataframe

# Handle missing values
data = data.dropna()

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in ['Cloud Cover', 'Season', 'Weather Type', 'Location']:
    data[col] = le.fit_transform(data[col])

print(data)

y = data['Temperature']
X = data.drop(columns=['Temperature'])
print(X.head())

# Split data into training and validation sets
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

# Print first 5 rows of scaled training data
print(X_train[:5])  

# Scale target variable
y_scaler = MinMaxScaler()
y_train = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_valid = y_scaler.transform(y_valid.values.reshape(-1, 1)).flatten()

# Define the Keras model
def create_model():
    model = Sequential()
    model.add(Dense(12, activation='relu', input_shape=[X_train.shape[1]]))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    adam = Adam(0.001)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mae', 'mse'])
    return model

model = create_model()
print(model.summary())

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=32,  # Adjusted batch size
    epochs=100,
    verbose=2,  # Changed verbosity to 2 for clearer output
)

# Plot training history
history_df = pd.DataFrame(history.history)

plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
history_df[['loss', 'val_loss']].plot(ax=plt.gca())
plt.title('Loss')

# Plot MAE
plt.subplot(1, 2, 2)
history_df[['mae', 'val_mae']].plot(ax=plt.gca())
plt.title('Mean Absolute Error')

plt.savefig('/home/georgikatevski/AI/WeatherCast/prediction_task2.png')
plt.close()

# Print best validation metrics
print("Best Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))
print("Best Validation MAE: {:0.4f}".format(history_df['val_mae'].min()))
print("Best Validation MSE: {:0.4f}".format(history_df['val_mse'].min()))

print(history_df.head())
