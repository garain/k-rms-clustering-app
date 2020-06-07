"""Flask App Project."""

from flask import Flask, jsonify
from krms_iris import *

app = Flask(__name__)


@app.route('/')  
def upload():  
    return render_template("file_upload_form.html")  
 
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)  
        return render_template("success.html", name = f.filename)  
@app.route('/results')
def index():
    """Return homepage."""
    json_data = main()#{'Hello': 'World!'}
    #return jsonify(json_data)
    return jsonify(json_data)     

if __name__ == '__main__':
    app.run()
