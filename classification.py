import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('/home/georgikatevski/AI/WeatherCast/weather_classification_data.csv')

# Display the first few rows of the dataframe
print(data.head())  

# Handle missing values
data = data.dropna()

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in ['Cloud Cover', 'Season', 'Weather Type']:
    data[col] = le.fit_transform(data[col])

print(data)

# Transform data to numeric to enable further analysis
y = data['Location']
X = data.select_dtypes(include=[np.number])
print(X.head())

# Split data into training and validation sets
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X, y, test_size=0.2, random_state=0)

# Scale features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

# Print first 5 rows of scaled training data
print(X_train[:5])  

# One-hot encode the target variable
y_train_encoded = pd.get_dummies(y_train)
y_valid_encoded = pd.get_dummies(y_valid)

# Ensure y_valid_encoded has the same columns as y_train_encoded
y_valid_encoded = y_valid_encoded.reindex(columns=y_train_encoded.columns, fill_value=0)

# Define the Keras model
def create_model():
    model = Sequential()
    model.add(Dense(12, activation='relu', input_shape=[X_train.shape[1]]))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(y_train_encoded.shape[1], activation='softmax'))

    adam = Adam(0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

model = create_model()
print(model.summary())

# Train the model
history = model.fit(
    X_train, y_train_encoded,
    validation_data=(X_valid, y_valid_encoded),
    batch_size=32,  # Adjusted batch size
    epochs=250,
    verbose=2,  # Changed verbosity to 2 for clearer output
)

# Plot training history
history_df = pd.DataFrame(history.history)

plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
history_df[['loss', 'val_loss']].plot(ax=plt.gca())
plt.title('Loss')

# Plot accuracy
plt.subplot(1, 2, 2)
history_df[['accuracy', 'val_accuracy']].plot(ax=plt.gca())
plt.title('Accuracy')

plt.savefig('/home/georgikatevski/AI/WeatherCast/classification_task2.png')
plt.close()

# Print best validation metrics
print("Best Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))
print("Best Validation Accuracy: {:0.4f}".format(history_df['val_accuracy'].max()))

print(history_df.head())
