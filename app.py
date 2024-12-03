from flask import Flask, render_template, request, jsonify
import os
from PIL import Image
from utils import *
from hyper import CONTENT_IMAGE_FOLDER, STYLE_IMAGE_FOLDER

global models

app = Flask(__name__)

@app.route('/')
def introduce():
    return render_template('introduce.html')

@app.route('/main', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        file1 = request.files['style']
        file2 = request.files['content']
        
        if file1 and file2:
            method = request.form['method']
            if method in models.keys():
                style_image = Image.open(io.BytesIO(file1.read())).convert('RGB')
                content_image = Image.open(io.BytesIO(file2.read())).convert('RGB')
                style_tensor = models[method]['preprocess'](style_image)
                content_tensor = models[method]['preprocess'](content_image)
                output_tensor = models[method]['model'](content_tensor, style_tensor)
                output_image = tensor_to_pil(output_tensor[0])

                return render_template('result.html', content_image=img2str(content_image), style_image=img2str(style_image), output_image=img2str(output_image))

    return render_template('main.html')

@app.route('/api/content_images')
def get_content_images():
    try:
        images = [
            f"{CONTENT_IMAGE_FOLDER}/{img}" 
            for img in os.listdir(CONTENT_IMAGE_FOLDER) 
            if img.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        ]
        return jsonify(images)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/style_images')
def get_style_images():
    try:
        images = [
            f"{STYLE_IMAGE_FOLDER}/{img}" 
            for img in os.listdir(STYLE_IMAGE_FOLDER) 
            if img.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        ]
        return jsonify(images)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    models = load_model()
    app.run(debug=True)
