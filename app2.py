from flask import Flask, render_template, jsonify
import os

app = Flask(__name__)

# Thư mục chứa hình ảnh
CONTENT_IMAGE_FOLDER = os.path.join('static', 'images/content')
STYLE_IMAGE_FOLDER = os.path.join('static', 'images/style')

@app.route('/')
def index():
    # Render giao diện chính
    return render_template('introduce.html')

@app.route('/main')
def main():
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
