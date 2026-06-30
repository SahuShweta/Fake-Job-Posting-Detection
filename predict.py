import joblib

# Load model and vectorizer
model = joblib.load("model/fake_job_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Sample job posting
job_post = """
Google is hiring a Software Engineer.
Experience in Python, Java, Data Structures,
Algorithms, Git, Linux and Cloud.
"""

# Convert text to TF-IDF
job_vector = vectorizer.transform([job_post])

# Predict
prediction = model.predict(job_vector)

if prediction[0] == 1:
    print("🚨 Fake Job Posting")
else:
    print("✅ Real Job Posting")