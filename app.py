
from flask import Flask, render_template, request
from train_model import get_model

app = Flask(__name__)

# Load the trained Machine Learning model
model = get_model()


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    email_text = ""

    if request.method == "POST":

        email_text = request.form.get("email_text", "").strip()

        if email_text:
            prediction = model.predict([email_text])[0]

            if prediction == "spam":
                result = "Spam"
            else:
                result = "Ham"

    return render_template(
        "index.html",
        result=result,
        email_text=email_text
    )


if __name__ == "__main__":
    app.run(debug=True)
