from flask import Flask, render_template, request, send_from_directory, jsonify
import os
from TransferModel.model import transfer

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
            suffix = len(os.listdir(app.config['UPLOAD_FOLDER']))

            style_path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], file2.filename)
            file1.save(style_path)
            file2.save(content_path)
            
            output_filename = f'output_image.jpg'
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            transfer(content_path, style_path, output_path)  # Gọi hàm style transfer

            return render_template('result.html', output_image=output_filename)

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
