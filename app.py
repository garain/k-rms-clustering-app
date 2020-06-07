"""Flask App Project."""

from flask import *
import os
#Flask, jsonify, render_template
from krms_iris import *

app = Flask(__name__)
app.config['/']
#File=str()
@app.route('/')  
def upload():  
    return render_template("file_upload_form.html")  
 
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)  
        #File=f.filename
        return render_template("success.html", name = f.filename)  
@app.route('/results', methods = ['POST'])
def index():
    """Return homepage."""
    if request.method == 'POST':  
        files = os.listdir("/")    
        files = list(filter(lambda f: f.endswith('.csv'), files))
        json_data = main(files[0])#{'Hello': 'World!'}
        #return jsonify(json_data)
        return jsonify(json_data)     

if __name__ == '__main__':
    app.run()
