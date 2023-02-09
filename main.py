# Importing packages
from detecto import core, utils, visualize
from flask import Flask, escape, request, jsonify, Response
import skimage
from flask_cors import CORS
from PIL import Image
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io

# Initilaising app and wrapping it in CORS to allow request from different services
app = Flask(__name__)
CORS(app)
# Telling matplotlib to not create GUI Windows as our application is backend and doesn't require direct visulaization
matplotlib.use('agg')
# Loading our custom model
model = core.Model.load('model_weights-2.pth', ['rbc'])


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Adding new POST endpoint that will accept image and output image with bounding boxes of detected objects
@app.route("/detect", methods=['POST'])
def detect():
    # Accessing file from request
    file = request.files['image']

    # Check if the file extension is allowed
    if not allowed_file(file.filename):
        return "Only PNG and JPG image formats are allowed", 400

    image = Image.open(file).convert('RGB')
    torch_model = model.get_internal_model()
    if torch_model.roi_heads.detections_per_img == 100:
        torch_model.roi_heads.detections_per_img = 1000
    # Using model to detect objects
    predictions = model.predict(image)
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
    ax.imshow(image)
    # Adding bounding boxes
    for i in range(len(box)):
        ax.add_patch(
            patches.Rectangle((box[i][0], box[i][1]), box[i][2] - box[i][0], box[i][3] - box[i][1], linewidth=1,
                              edgecolor='r', facecolor='none'))

    # Preparing output
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Sending response as png image
    return Response(output.getvalue(), mimetype='image/png')


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000, threaded=True)
