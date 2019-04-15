from flask import Flask, render_template, send_from_directory, jsonify, json
import os

spa_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'spa/dist/')
app = Flask(__name__, template_folder=spa_directory)

example_list = [
    {
        'key': 'harm-osc-2d',
        'name': '2D Harmonic Oscillator'
    }
]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/examples')
def examples():
    return jsonify(example_list)


@app.route('/js/<path:path>', methods=['GET'])
def serve_js(path):
    return send_from_directory(spa_directory + '/js', path)


@app.route('/css/<path:path>', methods=['GET'])
def serve_css(path):
    return send_from_directory(spa_directory + '/css', path)


@app.route('/img/<path:path>', methods=['GET'])
def serve_img(path):
    return send_from_directory(spa_directory + '/img', path)


if __name__ == "__main__":
    app.config['ENV'] = 'development'
    app.config['DEBUG'] = True
    app.run(host='0.0.0.0')
