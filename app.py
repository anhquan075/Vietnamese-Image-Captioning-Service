from flask import Flask
from flask_restful import Api

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload_images'
api = Api(app)