import base64
import datetime
import StringIO
import random

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
from io import BytesIO

from flask import Flask, request, render_template, Response
from werkzeug.utils import secure_filename
import urllib



from scanner_sparse import apply_model


ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'msp'])

app = Flask(__name__)

preds = None

@app.route("/")
def hello():
    return render_template('index.html')


@app.route("/index", methods=['POST'])
def index():
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
        print(type(file))
        print(dir(file))
        #print(file.readlines())
        png_output,preds =apply_model(file,"static/res/model.pickle","static/res/selected_features.txt",filename)
        figfile = BytesIO()
        plt.savefig(figfile, format='png')
        figfile.seek(0)
        figdata_png = base64.b64encode(figfile.getvalue())
        plt.close()
        #generate_large_csv(preds)
        #return figdata_png
        #png_output = simple()
        #return render_template("index.html", img_data=urllib.quote(png_output))
        return render_template("index.html", img_data=urllib.quote(figdata_png.rstrip('\n')))


@app.route('/test.csv')
def generate_large_csv():
    global preds
    print(",".join(preds.to_string().replace("\n","\\n").split()).replace("\\n","\n"))
    #csv_ret = "\n".join([",".join(map(str, row.insert(0,preds.index[index_row]))) for index_row,row in enumerate(preds.values.tolist())])
    return Response(",".join(preds.to_string().replace("\n","\\n").split()).replace("\\n","\n"), mimetype='text/csv', headers={"Content-Disposition":"attachment;filename=test.csv"})


def simple():
    fig=Figure()
    ax=fig.add_subplot(111)
    x=[]
    y=[]
    now=datetime.datetime.now()
    delta=datetime.timedelta(days=1)
    for i in range(10):
        x.append(now)
        now+=delta
        y.append(random.randint(0, 1000))
    ax.plot_date(x, y, '-')
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    canvas=FigureCanvas(fig)
    png_output = StringIO.StringIO()
    canvas.print_png(png_output)
    png_output = png_output.getvalue().encode("base64")

    return png_output


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == "__main__":
    app.run(debug=True)
