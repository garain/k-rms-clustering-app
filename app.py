"""Flask App Project."""

from flask import Flask, jsonify
from krms_iris import *

app = Flask(__name__)


@app.route('/')
def index():
    """Return homepage."""
    json_data = main()#{'Hello': 'World!'}
    #return jsonify(json_data)
    return jsonify(json_data)
    

if __name__ == '__main__':
    app.run()
