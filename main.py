from flask import Flask, render_template, request, redirect, url_for
import os
from detect_scc import detect_scc  # Import the SCC detection function


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_image():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    # Save the uploaded image to the upload folder
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Perform SCC detection
    result = detect_scc(file_path)

    return render_template("result.html", result=result, image_url=file_path)

if __name__ == "__main__":
    app.run(debug=True)
