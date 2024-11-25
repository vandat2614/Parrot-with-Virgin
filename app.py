from flask import Flask, render_template, request, send_from_directory, jsonify
import os
from TransferModel.adaattn_model import adaattn_transfer
from TransferModel.adain_model import adain_transfer
from PIL import Image
import io
import base64

app = Flask(__name__)

CONTENT_IMAGE_FOLDER = os.path.join('static', 'images/content')
STYLE_IMAGE_FOLDER = os.path.join('static', 'images/style')


def img2str(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

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

            if method == 'AdaIN':
                transfer = adain_transfer
            elif method == 'AdaAttN':
                transfer = adaattn_transfer

            style_image = Image.open(io.BytesIO(file1.read())).convert('RGB')
            content_image = Image.open(io.BytesIO(file2.read())).convert('RGB')
            output_image = transfer(content_image, style_image)  # Gọi hàm style transfer
            # output_image = content_image

            return render_template('result.html', content_image=img2str(content_image), style_image=img2str(style_image), output_image=img2str(output_image))

    return render_template('main.html')

@app.route('/api/content_images')
def get_content_images():
    # Lấy danh sách hình ảnh từ thư mục
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
    # Lấy danh sách hình ảnh từ thư mục
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
    app.run(debug=True)
