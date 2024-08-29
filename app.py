from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from pose_analyzer_class import PoseAnalyzer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

pose_analyzer = PoseAnalyzer()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        output_path = pose_analyzer.get_output_path(filepath)
        joints_to_process = request.form.getlist('joints')

        success, message = pose_analyzer.process_file(filepath, output_path, joints_to_process)

        if success:
            return jsonify({'message': message, 'output_path': output_path})
        else:
            return jsonify({'error': message})

@app.route('/analyze', methods=['POST'])
def analyze_data():
    selected_angles = request.json.get('selected_angles', [])
    success, message = pose_analyzer.analyze_data(selected_angles)
    if success:
        return jsonify({'message': message})
    else:
        return jsonify({'error': message})

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)