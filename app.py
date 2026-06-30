from flask import Flask, render_template, request
import joblib

# Create Flask application
app = Flask(__name__)

# Load trained model
model = joblib.load("model/fake_job_model.pkl")

# Load TF-IDF vectorizer
vectorizer = joblib.load("model/vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    title = request.form["title"]
    company = request.form["company"]
    description = request.form["description"]
    requirements = request.form["requirements"]
    benefits = request.form["benefits"]

    text = (
        title + " " +
        company + " " +
        description + " " +
        requirements + " " +
        benefits
    )

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)

    if prediction[0] == 1:
        result = "❌ Fake Job Posting"
    else:
        result = "✅ Real Job Posting"

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)