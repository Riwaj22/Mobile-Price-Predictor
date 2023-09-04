from flask_cors import cross_origin, CORS
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
cors = CORS(app)

data = pd.read_csv('final_mobile_data.csv')

# Load the pickled model
with open('mobile_trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

data['Processor'] = data['Processor'].astype(str)
data['RAM'] = data['RAM'].astype(float)

@app.route('/')
@app.route('/')
def index():
    Processor = sorted(data['Processor'].unique())
    rating = sorted(data['Rating ?/5'])
    no = sorted(data['Number of Ratings'])
    RAM = sorted(data['RAM'])
    ROM = sorted(data['ROM/Storage'])
    f = sorted(data['Front Camera'])
    Battery = sorted(data['Battery'])
    cam = sorted(data['Camera Resolutions'])
    ty = sorted(data['Camera Types'].unique())
    name = sorted(data['Name of Phone'].unique())
    # Fetch and format processor options based on phone names
    # processor_options = {}
    # for phone_name in name:
    #     processors = sorted(data[data['Name of Phone'] == phone_name]['Processor'].unique())
    #     processor_options[phone_name] = processors

    # Pass the processor_options variable to the template
    return render_template('index.html',
                           Processor=Processor,
                           name= name,
                           rating=rating,
                           no=no,
                           RAM=RAM,
                           ROM=ROM,
                           f=f,
                           Battery=Battery,
                           cam=cam,
                           ty=ty,
                           )



# @app.route('/get_processors', methods=['GET'])
# @cross_origin()
# def get_processors():
#     selected_name = request.args.get('name')
#     if selected_name:
#         processor_options = data[data['Name of Phone'] == selected_name]['Processor'].unique()
#         return jsonify(list(processor_options))
#     else:
#         return jsonify([])  # Return an empty list if no name is provided



@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        # Get form data
        Processor = request.form.get('Processor')
        rating = float(request.form.get('rating'))
        no = int(request.form.get('no'))  # Cast to int if it's an integer
        RAM = float(request.form.get('RAM'))  # Cast to float
        ROM = float(request.form.get('ROM'))  # Cast to float
        f = float(request.form.get('f'))  # Cast to float
        Battery = float(request.form.get('Battery'))  # Cast to float
        cam = float(request.form.get('cam'))  # Cast to float
        ty = request.form.get('ty')
        name = request.form.get('name')

        # Create a dictionary with the feature names and values
        data_dict = {
            'Processor': [Processor],
            'Rating ?/5': [rating],
            'Number of Ratings': [no],
            'RAM': [RAM],
            'ROM/Storage': [ROM],
            'Front Camera': [f],
            'Battery': [Battery],
            'Camera Resolutions': [cam],
            'Camera Types': [ty],
            'Name of Phone': [name]
        }

        # Create a DataFrame
        input_data = pd.DataFrame(data_dict)

        # Make the prediction
        prediction = model.predict(input_data)

        return jsonify({'prediction': prediction[0]})  # Send the prediction as JSON response
    except Exception as e:
        # Log the error
        app.logger.error(f'Prediction error: {str(e)}')

        # Return an error response
        return jsonify({'error': 'Prediction not available'}), 500



if __name__ == "__main__":
    app.run(debug=True)




