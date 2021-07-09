from flask import Flask
from app.config import Config
from flask_cors import CORS

app = Flask(__name__, template_folder="templates", static_folder='static')
app.config.from_object(Config)

CORS(app)

from app.sample_route.sample_blueprint import sample_blueprint
app.register_blueprint(sample_blueprint, url_prefix='/sample_route')

from app import routes