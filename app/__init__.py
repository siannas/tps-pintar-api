from flask import Flask
from app.config import Config
from flask_cors import CORS

app = Flask(__name__, template_folder="templates", static_folder='static')
app.config.from_object(Config)

CORS(app)

from app.face_blueprint import face_blueprint
app.register_blueprint(face_blueprint, url_prefix='/recognition')

from app import routes