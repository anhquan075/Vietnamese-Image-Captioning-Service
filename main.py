from app import api, app, basedir
from flask_restful import Resource, reqparse
import werkzeug

import os
import math

import numpy as np 
import cv2
import tensorflow as tf

from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary
from im2txt import configuration
from im2txt import inference_wrapper

from utils.parser import get_config
cfg = get_config()
cfg.merge_from_file('cfg/service.yml')

MODEL_PATH = cfg.SERVICE.CHECKPOINT_PATH
SERVICE_IP = cfg.SERVICE.SERVICE_IP
SERVICE_PORT = cfg.SERVICE.SERVICE_PORT
VOCAB_FILE = cfg.SERVICE.VOCAB_FILE

# Build the inference graph.
g = tf.Graph()
with g.as_default():
	model = inference_wrapper.InferenceWrapper()
	restore_fn = model.build_graph_from_config(configuration.ModelConfig(), MODEL_PATH)
g.finalize()

# Create the vocabulary.
vocab = vocabulary.Vocabulary(VOCAB_FILE)

ALLOWED_EXTENSIONS = set(['jpg', 'png', 'jpeg'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class serviceImageCaptioningHandler(Resource):
	def post(self):
		parser = reqparse.RequestParser()
		parser.add_argument('image', type=werkzeug.datastructures.FileStorage, location='files') 

		args = parser.parse_args()
		try:
			image_filename = args['image'].filename
			print(image_filename)
			if allowed_file(image_filename):
				image_file = args['image'].read()
				npimg = np.fromstring(image_file, np.uint8)
				img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
				cv2.imwrite(os.path.join(basedir, app.config['UPLOAD_FOLDER'], image_filename), img)

				with tf.Session(graph=g) as sess:
					# Load the model from checkpoint.
					restore_fn(sess)

					# Prepare the caption generator. Here we are implicitly using the default
					# beam search parameters. See caption_generator.py for a description of the
					# available beam search parameters.
					generator = caption_generator.CaptionGenerator(model, vocab, beam_size=1)
					
					with tf.gfile.GFile(os.path.join(basedir, app.config['UPLOAD_FOLDER'], image_filename), "rb") as f:
						image = f.read()
					basename = os.path.basename(image_filename)
					# dirname = os.path.dirname(image_filename)
					
					captions = generator.beam_search(sess, image)
					
					AllCaptions = []
					print(" Captions for image %s:" % (basename))
					for i, caption in enumerate(captions):
						# Ignore begin and end words.
						sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
						sentence = " ".join(sentence)
						AllCaptions.append(sentence)
					
					print(AllCaptions)

				return {'message': 'Successfully', 'caption': AllCaptions }, 200
		except:
			return {'message': 'No file selected'}, 419

		return {'message': 'Not in allowed file'}, 420


api.add_resource(serviceImageCaptioningHandler, '/api/image_captioning')
if __name__ == "__main__":
	app.run(host=SERVICE_IP, debug=True, port=SERVICE_PORT)
