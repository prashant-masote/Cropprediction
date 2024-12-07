
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

VALID_RANGES = {
    "Nitrogen": (0, 140),
    "Phosphorus": (5, 145),
    "Potassium": (5, 205),
    "temperature": (8.83, 43.68),
    "humidity": (14.26, 99.98),
    "pH": (3.50, 9.94),
    "rainfall": (20.21, 298.56),
}

def validate_input(input_data):
    """
    Validate the input values against predefined ranges.
    """
    for key, value in zip(VALID_RANGES.keys(), input_data):
        min_val, max_val = VALID_RANGES[key]
        if not (min_val <= value <= max_val):
            return f"Error: {key} value {value} is out of range ({min_val}-{max_val})."
    return None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect input data
        float_features = [float(x) for x in request.form.values()]
        input_data = np.array(float_features)

        # Validate input data
        validation_error = validate_input(float_features)
        if validation_error:
            return render_template("index.html", prediction_text=validation_error)

        # Make prediction
        prediction = model.predict([input_data])
        predicted_crop = prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction

        return render_template("index.html", prediction_text=f"The Predicted Crop is: {predicted_crop}")

    except ValueError:
        return render_template("index.html", prediction_text="Error: Invalid input. Please enter numeric values.")
    except Exception as e:
        return render_template("index.html", prediction_text=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
