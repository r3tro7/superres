# app.py
from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for
from utils import process_image
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # if file:
        #     image_path = os.path.join('static/images', file.filename)
        #     file.save(image_path)
        #     output_path = process_image(image_path)
        #     return render_template('index.html', output_image=output_path)
        if file:
            # Save the uploaded image
            image_path = os.path.join('static/images', file.filename)
            file.save(image_path)

            # Process the image and get the output path
            output_path = process_image(image_path)

            # Generate web-accessible paths
            web_image_path = url_for('static', filename='images/' + file.filename)
            web_output_path = url_for('static', filename='output/' + os.path.basename(output_path))

            # Render template with both image paths
            return render_template('index.html', input_image=web_image_path, output_image=web_output_path)

    return render_template('index.html')

@app.route('/output/<path:filename>')
def output(filename):
    return send_from_directory('static/output', filename)

if __name__ == '__main__':
    app.run(debug=True)