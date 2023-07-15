import base64
import datetime
from google.cloud import aiplatform as aip
from flask import Flask, jsonify, request, send_file, render_template
from flask_cors import CORS
from PIL import Image
from io import BytesIO

app = Flask(__name__)
CORS(app)

@app.route("/")
def root():
    # For the sake of example, use static information to inflate the template.
    # This will be replaced with real information in later steps.
    dummy_times = [
        datetime.datetime(2018, 1, 1, 10, 0, 0),
        datetime.datetime(2018, 1, 2, 10, 30, 0),
        datetime.datetime(2018, 1, 3, 11, 0, 0),
    ]

    return render_template("index.html", times=dummy_times)

@app.route('/predict', methods=['POST'])
def predict():
  PROJECT_NAME = 'story-ai-2'
  REGION = 'us-central1'
  ENDPOINT_ID = '7379917647586000896'
  
  # Get the input data from the HTTP request
  input_data = request.get_json()

  # Extract the text parameter from the input data
  prompt = input_data.get('prompt', '')

  aip.init(project=PROJECT_NAME, location=REGION)
  endpoint = aip.Endpoint(endpoint_name=ENDPOINT_ID)
  text_input = prompt

  # Invoke the Vertex AI
  payload = {"prompt": text_input}
  response = endpoint.predict(instances=[payload])
  
  # Decode the image data from base64 format
  image_data = response.predictions[0]
  image_bytes = base64.b64decode(image_data)

  # Create a PIL Image object from the decoded image data
  image = Image.open(BytesIO(image_bytes))

  # Save the image to a BytesIO buffer
  buffer = BytesIO()
  image.save(buffer, format='JPEG')
  buffer.seek(0)

  # Return the image file in the response
  return send_file(buffer, mimetype='image/jpeg')

if __name__ == "__main__":
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    # Flask's development server will automatically serve static files in
    # the "static" directory. See:
    # http://flask.pocoo.org/docs/1.0/quickstart/#static-files. Once deployed,
    # App Engine itself will serve those files as configured in app.yaml.
    app.run(host="127.0.0.1", port=8080, debug=True)
