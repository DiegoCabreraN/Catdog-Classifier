import os

from model import predict
from flask import Flask, request, url_for, render_template, redirect


app = Flask(__name__)

app.config['UPLOAD_PATH'] = 'static/img/'

last_file = ''

@app.route('/')
def index():
    global last_file
    if os.path.exists(last_file):
        os.remove(last_file)
    last_file = ''
    return render_template('index.html')

@app.route('/upload', methods=['GET','POST'])
def upload():
    global last_file
    if request.method == 'POST':

        sample_file = request.files['sample']
        last_file = os.path.join(app.config['UPLOAD_PATH'], sample_file.filename)
        sample_file.save(last_file)
    return render_template('upload.html', last_file=last_file)

@app.route('/restart')
def restart():
    global last_file
    if os.path.exists(last_file):
        os.remove(last_file)
    last_file = ''
    return redirect('/upload')

@app.route('/classifier')
def classifier():
    return render_template('classifier.html', prediction = predict(last_file), last_file = last_file)

app.run(debug=False)