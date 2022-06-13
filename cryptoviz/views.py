from flask import render_template, request
from cryptoviz import flaskapp
from cryptoviz.util import getNews, create_plot, update_dataset


@flaskapp.route('/news')
def news():
    return render_template('news.html', context=getNews('cryptocurrency'))


@flaskapp.route('/aboutus')
def about():
    return render_template("about.html")


@flaskapp.route('/predict')
def predict():
    return render_template("prediction.html")


@flaskapp.route('/predictGraph', methods=['POST', 'GET'])
def predictGraph():
    if request.method == "POST":
        print(request.form.get('pred_length'))
        length = int(request.form.get('pred_length'))
        update_dataset()
        p_url = create_plot(1000, length)
        return render_template("prediction-graph.html", plot_url='data:image/png;base64,{}'.format(p_url))


@flaskapp.route('/contactus')
def contact():
    return render_template("contact.html")


@flaskapp.route("/")
def home():
    return render_template("index.html")
