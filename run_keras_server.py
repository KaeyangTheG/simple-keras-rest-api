# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.models import load_model
from PIL import Image
from utils.utils import get_yolo_boxes
from utils.bbox import get_box_data
import numpy as np
import flask
import io
import cv2
import tensorflow as tf

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def load_api_model():
	global infer_model
	global graph
	infer_model = load_model("soda_can.h5")
	graph = tf.get_default_graph()

def load_variables():
	global anchors
	global net_h
	global net_w
	global obj_thresh
	global nms_thresh
	global labels

	anchors = [ 27, 63, 56, 198, 80, 111, 97, 252, 136, 326, 150, 130,
	191, 351, 253, 357, 381, 384]
	net_h, net_w = 416, 416
	obj_thresh, nms_thresh = 0.5, 0.45
	labels = ['soda_can']

@app.route("/predict", methods=["POST"])
def predict():
	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))

			open_cv_image = np.array(image)
			open_cv_image = open_cv_image[:, :, ::-1].copy()

			with graph.as_default():
				boxes = get_yolo_boxes(infer_model, [open_cv_image],
				net_h, net_w, anchors, obj_thresh, nms_thresh)[0]

				data = get_box_data(boxes, labels, obj_thresh)

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_variables()
	load_api_model()
	app.run()
