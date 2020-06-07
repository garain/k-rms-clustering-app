"""Flask App Project."""

from flask import Flask, jsonify
from krms_iris import *

app = Flask(__name__)


@app.route('/')
def index():
    """Return homepage."""
    #json_data = {'Hello': 'World!'}
    #return jsonify(json_data)
    return main()
    

if __name__ == '__main__':
    app.run()
