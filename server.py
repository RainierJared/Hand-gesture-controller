from flask import Flask, render_template, Response, send_file
from camera import VideoCamera

app = Flask(__name__)
app._static_folder = 'static'

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame
              + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/manifest.json')
def serve_manifest():
    return send_file('manifest.json', mimetype='statitc/manifest+json')

@app.route('/sw.js')
def serve_sw():
    return send_file('sw.js', mimetype='application/javascript')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ =="__main__":
    app.run(host='172.20.10.2', port='8000', debug=True)