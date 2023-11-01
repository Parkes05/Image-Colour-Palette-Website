from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap5
from werkzeug.utils import secure_filename
import numpy as np
import cv2, os
from sklearn.cluster import KMeans


def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_KEY')
Bootstrap5(app)

upload = os.path.join('static', 'uploads')
app.config['UPLOAD'] = upload


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        f = request.files['img']
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD'], filename))
        my_img = os.path.join(app.config['UPLOAD'], filename)

        img = cv2.imread(my_img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = img.reshape((img.shape[1]*img.shape[0],3))
        kmeans = KMeans(n_clusters=10)
        kmeans.fit(img)
        centroid=kmeans.cluster_centers_
        colors=np.array(centroid)

        colors = colors.tolist()
        hex_list = []
        for i in colors:
            hex_list.append(rgb_to_hex(int(i[0]), int(i[1]), int(i[2])))
        return render_template('index.html', src=my_img, colors=hex_list)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
