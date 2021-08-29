from app import api, app
from flask_restful import Resource, reqparse
import werkzeug
import os
import numpy as np 
import cv2

ALLOWED_EXTENSIONS = set(['jpg', 'png', 'jpeg'])
basedir = os.path.abspath(os.path.dirname(__file__))
os.makedirs('upload_images', exist_ok=True)


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class serviceImageCaptioningHandler(Resource):
	def post(self):
		parser = reqparse.RequestParser()
		parser.add_argument('image', type=werkzeug.datastructures.FileStorage, location='files') 

		args = parser.parse_args()

		try:
			image_filename = args['image'].filename
			if allowed_file(image_filename):
				image_file = args['image'].read()
				npimg = np.fromstring(image_file, np.uint8)
				img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)

				cv2.imwrite(os.path.join(basedir, app.config['UPLOAD_FOLDER'], image_filename), img)
				return {'message' : 'Sent image successfully'}, 200
		except:
			return {'message' : 'No file selected'}, 300

		return {'message': 'Not in allowed file'}, 202 

api.add_resource(serviceImageCaptioningHandler, '/api/image_captioning')
if __name__ == "__main__":
	app.run(host='0.0.0.0', debug=True, port=5000)