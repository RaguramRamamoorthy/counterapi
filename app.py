# Importing packages
# from detecto import core, utils, visualize
import PIL.Image
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
import cv2

# Initilaising app and wrapping it in CORS to allow request from different services
app = Flask(__name__)
CORS(app)
# Telling matplotlib to not create GUI Windows as our application is backend and doesn't require direct visulaization
matplotlib.use('agg')


# Loading our custom model
# model = core.Model.load('model_weights.pth', ['rbc'])
def crop_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # (2) Morph-op to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    # (3) Find the max-area contour
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    # (4) Crop and save it
    x, y, w, h = cv2.boundingRect(cnt)
    dst = image[y:y + h, x:x + w]
    return dst


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
@app.route("/bnw", methods=['POST'])
def bnw():
    data = request.json
    image_rec = data['image']
    slider = data['slider']

    # Decode the Base64-encoded image data to bytes
    image_bytes = base64.b64decode(image_rec)
    image_byte = io.BytesIO(image_bytes)

    # Open the image data as an Image object
    imagebnw = Image.open(image_byte)
    converted_img = np.array(imagebnw)
    gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(gray_scale, slider, 255, cv2.THRESH_BINARY)
    image = Image.fromarray(blackAndWhiteImage)

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')

    base64_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    # converted_img = np.array(imagebnw)
    # gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
    # (thresh, blackAndWhiteImage) = cv2.threshold(gray_scale, slider, 255, cv2.THRESH_BINARY)

    data = {
        'image': base64_image
        # 'image': base64.b64encode(black_and_white_image).decode('utf-8')
    }

    # Convert the dictionary to JSON
    json_data = json.dumps(data)

    # Return the JSON response
    return Response(json_data, mimetype='application/json')


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
    image_with_count = output.getvalue()

    image = Image.open(io.BytesIO(image_with_count))

    converted_img = np.array(image.convert('RGB'))
    image_cropped = crop_image(converted_img)
    image_pil = Image.fromarray(image_cropped)

    output1 = io.BytesIO()
    # Save the image to the BytesIO object
    image_pil.save(output1, format='PNG')
    # Get the byte stream value
    image_bytes = output1.getvalue()

    number = len(labels)
    print(number)

    data = {
        'image': base64.b64encode(image_bytes).decode('utf-8'),
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


print('hi')