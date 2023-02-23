# Importing packages
# from detecto import core, utils, visualize
from flask import Flask, escape, request, jsonify, Response
from flask_cors import CORS
from PIL import Image
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import numpy as np
import json
import base64
from load import *

# Initilaising app and wrapping it in CORS to allow request from different services
app = Flask(__name__)
CORS(app)
# Telling matplotlib to not create GUI Windows as our application is backend and doesn't require direct visulaization
matplotlib.use('agg')


# Loading our custom model
# model = core.Model.load('model_weights.pth', ['rbc'])


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def hello_world():
    return 'Hello!'


# Adding new POST endpoint that will accept image and output image with bounding boxes of detected objects
@app.route("/test", methods=['POST'])
def test():
    # Accessing file from request
    file = request.files['image']

    # Check if the file extension is allowed
    if not allowed_file(file.filename):
        return "Only PNG and JPG image formats are allowed", 400

    image = Image.open(file).convert('RGB')
    x = np.invert(image)
    response = np.array_str(x)
    return response


# Adding new POST endpoint that will accept image and output image with bounding boxes of detected objects
@app.route("/detect", methods=['POST'])
def detect():
    model = init()
    # Accessing file from request
    # file = request.files['image']
    data = request.json
    image_rec = data['image']

    # Decode the Base64-encoded image data to bytes
    image_bytes = base64.b64decode(image_rec)
    image_byte = io.BytesIO(image_bytes)

    # Open the image data as an Image object
    image1 = Image.open(image_byte).convert('RGB')

    torch_model = model.get_internal_model()
    if torch_model.roi_heads.detections_per_img == 100:
        torch_model.roi_heads.detections_per_img = 1000
    # Using model to detect objects
    predictions = model.predict(image1)
    labels, boxes, scores = predictions
    # Applying threshold
    lab = []
    box = []
    for i in range(len(scores)):
        if scores[i] > 0.3:
            lab.append(labels[i])
            box.append(boxes[i])
    box = torch.stack(box)
    # Creating figure and displaying original image
    fig, ax = plt.subplots(1)

    # Set aspect ratio to equal the ratio of the image dimensions
    # aspect_ratio = image1.width / image1.height
    # ax.set_aspect(aspect_ratio)

    plt.axis('off')
    ax.imshow(image1)
    # Adding bounding boxes
    for i in range(len(box)):
        ax.add_patch(
            patches.Rectangle((box[i][0], box[i][1]), box[i][2] - box[i][0], box[i][3] - box[i][1], linewidth=1,
                              edgecolor='r', facecolor='none'))

    # fig.patch.set_alpha(0)
    #
    # # Set the DPI to the default value of 100
    # fig.set_dpi(100)

    # Preparing output
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    number = len(labels)

    data = {
        'image': base64.b64encode(output.getvalue()).decode('utf-8'),
        'number': number
    }

    # Convert the dictionary to JSON
    json_data = json.dumps(data)

    # Return the JSON response
    return Response(json_data, mimetype='application/json')

    # # Sending response as png image
    # return Response(output.getvalue(), mimetype='image/png')


if __name__ == '__main__':
    app.run()
