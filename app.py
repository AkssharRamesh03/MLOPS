from flask import Flask, render_template, request, jsonify
import os 
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline


app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 


@app.route('/predict', methods=['POST'])
def predict():

    try:
        # Get JSON data from the request
        data = request.json

        # Extract the features from the JSON data
        SALEQTY = int(data['SALEQTY'])
        SHOPCODE = int(data['SHOPCODE'])
        SUPPLIERCODE = int(data['SUPPLIERCODE'])
        CAT1CODE = int(data['CAT1CODE'])
        CAT3CODE = int(data['CAT3CODE'])
        PROFCODE = int(data['PROFCODE'])
        GROUPPROFCODE = int(data['GROUPPROFCODE'])

        # Create a NumPy array from the extracted features
        input_data = np.array([[
            SALEQTY, SHOPCODE, SUPPLIERCODE, CAT1CODE,
            CAT3CODE, PROFCODE, GROUPPROFCODE
        ]])

        obj = PredictionPipeline()
        prediction = np.array(obj.predict(input_data))
        print(prediction)

        # You can return predictions as JSON or perform further processing

        # Return a JSON response (example)
        response_data = {
            'message': 'Predictions successful',
            'predictions': prediction.tolist()  # Replace with your actual predictions
        }

        return jsonify(response_data), 200

    except Exception as e:
            # Handle exceptions or errors
            error_message = str(e)
            response_data = {'message': 'Error: ' + error_message}
            return jsonify(response_data), 400


if __name__ == "__main__":
	# app.run(host="0.0.0.0", port = 8080, debug=True)
	app.run(host="0.0.0.0", port = 8080)