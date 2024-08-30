import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load mock dataset
data = pd.read_csv('data/mock_stress_data.csv')
x_data = data['x'].values
y_data = data['y'].values
stress_xx = data['stress_xx'].values
stress_yy = data['stress_yy'].values
stress_xy = data['stress_xy'].values

# Normalize input data
X = np.vstack([x_data, y_data]).T

# Define the PINN model
class PINNModel(tf.keras.Model):
    def __init__(self):
        super(PINNModel, self).__init__()
        self.hidden1 = tf.keras.layers.Dense(20, activation='tanh')
        self.hidden2 = tf.keras.layers.Dense(20, activation='tanh')
        self.hidden3 = tf.keras.layers.Dense(20, activation='tanh')
        self.output_layer = tf.keras.layers.Dense(3)  # Outputs: stress_xx, stress_yy, stress_xy

    def call(self, inputs):
        x = self.hidden1(inputs)
        x = self.hidden2(x)
        x = self.hidden3(x)
        return self.output_layer(x)

# Initialize model
model = PINNModel()

# Define the custom loss function that incorporates physics laws (PDEs)
def custom_loss(y_true, y_pred):
    # Extract predicted stresses
    stress_xx_pred = y_pred[:, 0]
    stress_yy_pred = y_pred[:, 1]
    stress_xy_pred = y_pred[:, 2]

    # Physics-informed loss (e.g., Navier's equations)
    # Example: assuming simplified equations for demonstration
    physics_loss = tf.reduce_mean(tf.square(stress_xx_pred + stress_yy_pred))

    # Data loss
    data_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    return data_loss + physics_loss

# Compile model
model.compile(optimizer='adam', loss=custom_loss)

# Train the model
model.fit(X, np.vstack([stress_xx, stress_yy, stress_xy]).T, epochs=100, batch_size=16)

# Predict stress distribution
predicted_stresses = model.predict(X)

# Plot the predicted vs actual stress distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], predicted_stresses[:, 0], label='Predicted stress_xx', alpha=0.5)
plt.scatter(X[:, 0], stress_xx, label='Actual stress_xx', alpha=0.5)
plt.xlabel('x')
plt.ylabel('Stress (Pa)')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], predicted_stresses[:, 1], label='Predicted stress_yy', alpha=0.5)
plt.scatter(X[:, 0], stress_yy, label='Actual stress_yy', alpha=0.5)
plt.xlabel('x')
plt.ylabel('Stress (Pa)')
plt.legend()

plt.show()
