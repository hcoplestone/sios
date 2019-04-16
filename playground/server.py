from flask import Flask, render_template, send_from_directory, jsonify, abort, request
import os
import playground.examples as examples
import json

spa_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'spa/dist/')
app = Flask(__name__, template_folder=spa_directory)

example_list = [
    {
        'key': 'harm-osc-2d',
        'name': '2D Harmonic Oscillator',
        'description': 'A harmonic oscillator with two independent degrees of freedom oscillating in the x-y plane.',
        'params': [
            {
                'key': 't-lower',
                'order': 0,
                'name': 'Starting time',
                'value': 0
            },
            {
                'key': 't-upper',
                'order': 1,
                'name': 'Finish time',
                'value': 10
            },
            {
                'key': 'n',
                'name': 'Number of points',
                'value': 100
            },
            {
                'key': 'order-of-integrator',
                'name': 'Order of integrator',
                'value': 1
            },
            {
                'key': 'initial-x',
                'name': 'Initial x',
                'value': 0
            },
            {
                'key': 'initial-y',
                'name': 'Initial y',
                'value': 0
            },
            {
                'key': 'initial-x-momentum',
                'name': 'Initial x momentum',
                'value': 1
            },
            {
                'key': 'initial-y-momentum',
                'name': 'Initial y momentum',
                'value': 1
            }
        ]
    }
]

example_defs = {'harm-osc-2d': examples.two_dimension_harmonic_oscillator}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/examples')
def examples():
    return jsonify(example_list)


@app.route('/examples/<string:key>')
def get_example(key):
    example = next((example for example in example_list if example["key"].upper() == key.upper()), False)
    if not example:
        abort(404)
    else:
        return jsonify(example)


@app.route('/examples/<string:key>/integrate', methods=['POST'])
def integrate_example(key):
    example = next((example for example in example_list if example["key"].upper() == key.upper()), False)
    if not example:
        abort(404)
    else:
        return jsonify(example_defs[key](json.loads(request.data)))


@app.route('/js/<path:path>', methods=['GET'])
def serve_js(path):
    return send_from_directory(spa_directory + '/js', path)


@app.route('/css/<path:path>', methods=['GET'])
def serve_css(path):
    return send_from_directory(spa_directory + '/css', path)


@app.route('/img/<path:path>', methods=['GET'])
def serve_img(path):
    return send_from_directory(spa_directory + '/img', path)


def serve():
    app.config['ENV'] = 'development'
    app.config['DEBUG'] = True
    app.run(host='0.0.0.0')
