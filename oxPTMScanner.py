import base64

import matplotlib.pyplot as plt
from io import BytesIO

from flask import Flask, request, render_template, Response, send_file
from werkzeug.utils import secure_filename
import urllib


from oxPTM_scanner_web import apply_model


ALLOWED_EXTENSIONS = set(['mgf', 'msp'])

app = Flask(__name__)

preds = None


@app.route("/oxptmscanner/index")
@app.route("/")
def hello():
    return render_template('index.html')


@app.route("/oxptmscanner/index", methods=['POST'])
def index():
    error = None
    global preds
    # check if the post request has the file part
    if 'mgfInput' not in request.files:
        render_template('index.html')
    file = request.files['mgfInput']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        return render_template('index.html')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        png_output,preds =apply_model(file,"static/res/model.pickle","static/res/selected_features.txt",filename)
        figfile = BytesIO()
        plt.savefig(figfile, format='png')
        figfile.seek(0)
        figdata_png = base64.b64encode(figfile.getvalue())
        plt.close()
        return render_template("index.html", img_data=urllib.quote(figdata_png.rstrip('\n')))
    else:
        error = "Invalid input file type. Please upload .msp or .mgf file"
        return render_template("index.html", error=error)



@app.route('/result.csv')
def generate_large_csv():
    global preds
    return Response("Name,Probability oxPTM\n"+",".join(preds.to_string().replace("\n","\\n").split()).replace("\\n","\n"), mimetype='text/csv', headers={"Content-Disposition":"attachment;filename=result.csv"})


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/example.msp')
def return_files_tut():
    try:
        return send_file("static/NIST/example.msp", attachment_filename='example.msp')
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(debug=True)
