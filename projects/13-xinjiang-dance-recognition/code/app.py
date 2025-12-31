# app.py
from flask import Flask, request, jsonify, render_template
import os, pathlib, inference

app = Flask(__name__)
UPLOAD = 'uploads'
os.makedirs(UPLOAD, exist_ok=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('video')
        if not file or not file.filename.lower().endswith('.mp4'):
            return jsonify({'error': '请上传 .mp4 文件'}), 400

        save_path = os.path.join(UPLOAD, pathlib.Path(file.filename).name)
        file.save(save_path)

        result = inference.predict_video(save_path)
        os.remove(save_path)  # 清理上传
        return jsonify(result)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=63342, debug=True)