# Expert-Discovery-Manager

We will create a collection of mlp models, each with its own flask server and directory for storing reference data. Finally the goal is to apply all of them in parallel to solve a complex task.

In order to allow the models to discover themselves and to communicate with each other we need to implement some kind of discovery service.

### Discovery service

The discovery service provides a REST API for registering and deregistering models and allows to query the current active models.

```python
from flask import Flask, jsonify, request, abort

app = Flask(__name__)
models = {}

@app.route('/model', methods=['GET'])
def get_tasks():
    return jsonify({'models': models})


@app.route('/model', methods=['POST'])
def create_task():
    if not request.json or 'id' not in request.json:
        abort(400)

    models[request.json['id']] = {'url': request.json['url']}
    return jsonify({'models': models})


@app.route('/model/<id>', methods=['DELETE'])
def delete_task(id):
    if id not in models:
        abort(404)

    del models[id]
    return jsonify({'result': True})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
```

### Model service

The model service provides a REST API for training and testing a model based on given input data. We use the [scikit-learn](http://scikit-learn.org/stable/) implementation of a multi layer perceptron with [Huber loss](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html).

```python
from flask import Flask, request, jsonify
from sklearn.externals import joblib
from sklearn.linear_model import HuberRegressor
import numpy as np
import random
import os

app = Flask(__name__)
MODEL_DIR = 'models'

def load_model(filename):
    return joblib.load(os.path.join(MODEL_DIR, filename))

def train_model(data):
    model = HuberRegressor()
    model.fit(data['x'], data['y'])
    return model

def save_model(model, filename):
    joblib.dump(model, os.path.join(MODEL_DIR, filename))

@app.route('/train', methods=['POST'])
def train():
    if not request.json or not 'data' in request.json:
        abort(400)

    model = train_model(request.json['data'])
    filename = 'model-%d.pkl' % random.randint(0, 100000)
    save_model(model, filename)

    return jsonify({'filename': filename})

@app.route('/test', methods=['POST'])
def test():
    if not request.json or not 'data' in request.json:
        abort(400)

    model = load_model(request.json['filename'])
    y = model.predict(request.json['data']['x'])

    return jsonify({'y': y})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
```

### Create models

We create a script to start the discovery service and a number of model services. In order to ensure that each model is running on its own port we use the [Free Port Search](https://pypi.python.org/pypi/free-port-search) library.

```python
from multiprocessing import Process
import requests
import random
import os
from free_port_search import find_free_port
from flask import Flask, request, jsonify

def start_discovery_service():
    discovery_service.run(host='0.0.0.0', port=5000)

def start_model_service(name):
    model_service.run(host='0.0.0.0', port=find_free_port())
    url = 'http://localhost:5000/model'
    payload = {'id': name, 'url': 'http://localhost:%d/test' % model_service.port}
    requests.post(url, json=payload)

if __name__ == '__main__':
    discovery_service = Flask(__name__)
    @discovery_service.route('/')
    def home():
        return 'Discovery Service'

    p1 = Process(target=start_discovery_service)
    p1.start()

    for i in range(5):
        model_service = Flask('model-%d' % i)
        @model_service.route('/')
        def home():
            return 'Model-%d Service' % i
        p = Process(target=start_model_service, args=(i,))
        p.start()
```

### Train models

We create a script to generate some training data and to train the model services.
We assume that we have a 2d input space with x values between 0 and 1, y values between 0 and 10 and an error of up to 5.

```python
import requests
from sklearn.externals import joblib
from sklearn.linear_model import HuberRegressor
import numpy as np
import os

def generate_training_data(n):
    x = np.random.uniform(0, 1, (n, 1))
    y = x * 10 + np.random.normal(0, 5, (n, 1))

    return {'x': x, 'y': y}

def get_model_services():
    url = 'http://localhost:5000/model'
    r = requests.get(url)

    return r.json()['models']

def train_model_services():
    data = generate_training_data(100)

    for model_id in get_model_services():
        url = 'http://localhost:%s/train' % get_model_services()[model_id]['url']
        payload = {'data': data}
        r = requests.post(url, json=payload)
```

### Test models

We create a script to generate some test data and to query the model services for predictions. We assume that the test data is similar to the training data.

```python
import requests
from sklearn.externals import joblib
from sklearn.linear_model import HuberRegressor
import numpy as np
import os

def generate_test_data(n):
    x = np.random.uniform(0, 1, (n, 1))

    return {'x': x}

def get_model_services():
    url = 'http://localhost:5000/model'
    r = requests.get(url)

    return r.json()['models']

def test_model_services():
    data = generate_test_data(100)

    for model_id in get_model_services():
        url = 'http://localhost:%s/test' % get_model_services()[model_id]['url']
        payload = {'data': data}
        r = requests.post(url, json=payload)
```

### Aggregate results

We create a script to aggregate the results of the models and to calculate a mean prediction. We assume that we have 5 models available and that each model returns an array of predictions.

```python
import requests
from sklearn.externals import joblib
from sklearn.linear_model import HuberRegressor
import numpy as np
import os

def get_model_services():
    url = 'http://localhost:5000/model'
    r = requests.get(url)

    return r.json()['models']

def aggregate_model_services():
    for model_id in get_model_services():
        url = 'http://localhost:%s/test' % get_model_services()[model_id]['url']
        payload = {'data': data}
        r = requests.post(url, json=payload)

    y = np.mean(r.json()['y'])

    return y
```

### Parallel processing

In order to be able to apply multiple models in parallel we need to use a thread-safe flask server. We can do this by replacing the default Werkzeug WSGI Server with the [Gevent](http://www.gevent.org/) server. Please note that the Gevent server is not recommended for production use.

```python
from gevent.wsgi import WSGIServer
...
if __name__ == '__main__':
    app.debug = True
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
```

## Run the application

```bash
$ python start.py &> server-log.txt &
$ python train.py &> train-log.txt &
$ python test.py &> test-log.txt &
$ python aggregate.py &> aggregate-log.txt &
```

## References

[1] https://github.com/miguelgrinberg/flasky

[2] http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html

[3] https://pypi.python.org/pypi/free-port-search
