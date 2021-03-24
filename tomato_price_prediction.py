from flask import Flask, render_template, request
# import pickle
import numpy as np
from keras.models import model_from_json

app = Flask(__name__)

# Unpickling the ML model to use for prediction
# load json and create model
with open("MIMO_model.json", 'r') as json_file:
    loaded_json_file = json_file.read()

loaded_MIMO_model = model_from_json(loaded_json_file)
print('Loaded json file')

# load weights into newly loaded model
loaded_MIMO_model.load_weights("MIMO_model_wts.h5")
print("Loaded MIMO model from disk")
loaded_MIMO_model.compile(optimizer='adam', loss='mse')

print('New Model Summary')
print(loaded_MIMO_model.summary())

# For MIMO model , we are giving last 10 lag values as predictors.
previous_lag = np.array([1600, 1250, 800, 800, 862.5, 862.5, 787.5, 900., 900., 700.]).reshape(1, -1)
print(previous_lag.shape)


@app.route('/')
def home():
    return render_template('tomato_price.html')


@app.route("/info")
def info():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    inputs = [i for i in request.form.values()]
    # print(len(request.form.values()))
    print("State : ", inputs[0])
    print("District : ", inputs[1])
    print("Commodity : ", inputs[2])

    print("Prediction horizon : ", int(inputs[3]))

    horizon = int(inputs[3])

    predictions = loaded_MIMO_model.predict(previous_lag)[0]
    print()

    output = round(predictions[horizon], 3)
    print("Forecasted Output is : ", output)

    return render_template('tomato_price.html', prediction_text= f'Forecasted Value : {output}')


if __name__ == '__main__':
    app.run(debug=True)
