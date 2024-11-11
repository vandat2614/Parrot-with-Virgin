from flask import Flask, render_template, request, send_from_directory, session
import os
from TransferModel.model import transfer 

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file1 = request.files['style']
        file2 = request.files['content']
        
        if file1 and file2:
            suffix = len(os.listdir(app.config['UPLOAD_FOLDER']))

            style_path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], file2.filename)
            file1.save(style_path)
            file2.save(content_path)
            
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'output_image.jpg')
            transfer(content_path, style_path, output_path)  # Gọi hàm style transfer

            return render_template('result.html', output_image=output_path)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
