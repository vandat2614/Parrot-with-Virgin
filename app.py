from flask import Flask, render_template, request, send_from_directory, jsonify
import os
from TransferModel.model import transfer
from PIL import Image
import io
import base64

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
CONTENT_IMAGE_FOLDER = os.path.join('static', 'images/content')
STYLE_IMAGE_FOLDER = os.path.join('static', 'images/style')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Route chính cho trang giới thiệu
@app.route('/')
def introduce():
    return render_template('introduce.html')

# Route cho trang index, chuyển tiếp từ trang introduce
@app.route('/main', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        file1 = request.files['style']
        file2 = request.files['content']
        
        if file1 and file2:
            style_image = Image.open(io.BytesIO(file1.read())).convert('RGB')
            content_image = Image.open(io.BytesIO(file2.read())).convert('RGB')

            output_image = transfer(content_image, style_image)  # Gọi hàm style transfer

            buffered = io.BytesIO()
            output_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            return render_template('result.html', output_image=img_str)

    return render_template('main.html')

# Route để lấy file ảnh đã upload
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route để lấy file ảnh kết quả
@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

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
