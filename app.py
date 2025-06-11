from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Model setup
data = pd.read_csv("imdb.csv")
x = data.iloc[:, :-1]
y = data.iloc[:, -1]
if x.shape[1] == 1:
    x = x.squeeze()
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=5)
cv = CountVectorizer()
xtrain_vec = cv.fit_transform(xtrain)
model = MultinomialNB()
model.fit(xtrain_vec, ytrain)

# Read your HTML file manually
with open("index.html", "r", encoding="utf-8") as file:
    html_template = file.read()

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    if request.method == "POST":
        review = request.form.get("review", "")
        if review.strip():
            review_vec = cv.transform([review])
            prediction = model.predict(review_vec)
            sentiment = "Positive Review" if prediction[0] == 1 else "Negative Review"
    return render_template_string(html_template, sentiment=sentiment)
from flask import send_file

@app.route("/bgimage.jpg")
def image():
    return send_file("bgimage.jpg", mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True)
