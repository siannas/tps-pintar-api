from app import app
from flask import render_template, jsonify
from flask import abort, request, redirect, url_for

@app.route('/')
def home():
    return render_template('landing.html')