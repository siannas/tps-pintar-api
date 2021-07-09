from flask import abort, request, redirect, url_for, jsonify
from flask import Blueprint, render_template

from app.config import Config

sample_blueprint = Blueprint('sample_blueprint', __name__,
    template_folder='templates',
    static_folder='static', static_url_path='assets')

@sample_blueprint.route('/', methods=['GET','POST'])
def sample_view():
    return render_template('sample_route/sample_view.html')