"""Flask App Project."""

from flask import *
import os
#Flask, jsonify, render_template
from krms_iris import *

app = Flask(__name__)
#app.config['UPLOAD_FOLDER']="/"
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
        json_data = main(f.filename)
        session['data'] = json_data
        return render_template("success.html", name = f.filename)  
@app.route('/results', methods = ['POST'])
def index():
    """Return homepage."""
    if request.method == 'POST': 
        #{'Hello': 'World!'}
        #return jsonify(json_data)
        json_data=session.get('data', None)
        return jsonify(json_data)     

if __name__ == '__main__':
    app.run()
