from flask import Flask, request, render_template
import joblib
from scraper import Scrap
from predictor import Predictor

cv = joblib.load("CountVectorizer.pkl")
model = joblib.load("SentimentAnalyzer.pkl")

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/reviews", methods=['POST'])
def reviews():
    inpt = request.form['url']
    
    scraper = Scrap(url=inpt)
    reviews = scraper.scrap()
    
    predictor = Predictor()
    pos, neg = predictor.predict(reviews, cv, model)
  
    return render_template('index.html',tables=[pos.to_html(classes='positive'), neg.to_html(classes='negative')],
    titles = ['na', 'Most Positive Reviews', 'Most Negative Reviews'])


if __name__ == "__main__":
    app.run()