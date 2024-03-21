from flask import Flask, render_template, request
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from joblib import load as joblib_load


app = Flask(__name__, template_folder='template')


# Load your LSTM model
model_lstm = load_model('model.h5')

# Load other models if needed
# ...
# Load your RandomForestRegressor model
rf_model = joblib_load('rf_model.joblib')

# Load your SVR model
svr_model = joblib_load('svr_model.joblib')

# Load the MinMaxScaler (used during training)
scaler = MinMaxScaler(feature_range=(0, 1))

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input
        user_input = float(request.form['user_input'])

        # Normalize the user input using the same scaler used during training
        normalized_input = scaler.transform(np.array([[user_input]]).reshape(1, -1))

        # Make predictions using your LSTM model
        lstm_prediction = model_lstm.predict(normalized_input.reshape(1, 1, 1))
        lstm_prediction = scaler.inverse_transform(lstm_prediction.reshape(-1, 1))[0][0]

        # Make predictions using your RandomForestRegressor model
        rf_prediction = rf_model.predict(normalized_input)
        rf_prediction = scaler.inverse_transform(rf_prediction.reshape(-1, 1))[0][0]

        # Make predictions using your SVR model
        svr_prediction = svr_model.predict(normalized_input.reshape(1, -1))
        svr_prediction = scaler.inverse_transform(svr_prediction.reshape(-1, 1))[0][0]

        return render_template('index.html', 
                               lstm_prediction=lstm_prediction, 
                               rf_prediction=rf_prediction,
                               svr_prediction=svr_prediction)

if __name__ == '__main__':
    app.run(debug=False)