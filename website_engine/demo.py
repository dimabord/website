__author__ = 'dimabord'
from sentiment_analysis.sentiment_classifier import SentimentClassifier
from codecs import open
import time
from flask import Flask, render_template, request

app = Flask(__name__)

print ("Preparing classifier")
start_time = time.time()
classifier = SentimentClassifier()
print ("Classifier is ready")
print (time.time() - start_time, "seconds")

sentiment_site = 'sentiment.html'
cv_site = 'cv.html'
poetry_site = 'poetry.html'
main_site = 'main.html'

@app.route("/cv")
def cv():
    return render_template(poetry_site)

@app.route("/poetry")
def poetry():
    return render_template(poetry_site)

@app.route("/")
def main():
    return render_template(main_site)


@app.route("/sentiment", methods=["GET", "POST"])
def index_page(text="", prediction_message=""):
    logfile = open("ydf_demo_logs.txt", "a", "utf-8")
    if request.method== "POST":
        text = request.form["text"]
        prediction_message = classifier.get_prediction_message(text)
    print(text, file=logfile)
    print( "<response>", file=logfile)

    print(prediction_message, file=logfile)
    print("</response>", file=logfile)
    logfile.close()
    return render_template(sentiment_site, text=text, prediction_message=prediction_message)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=False)
