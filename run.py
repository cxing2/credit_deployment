from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle


with open("rfmodel.pkl", "rb") as f:
    rfmodel = pickle.load(f)

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    pred = ""
    category = ""
    if request.method == "POST":
        Credit_Mix = request.form["Credit_Mix"]
        Outstanding_Debt = request.form["Outstanding_Debt"]
        Interest_Rate = request.form["Interest_Rate"]
        Payment_of_Min_Amount = request.form["Payment_of_Min_Amount"]
        Num_Credit_Inquiries = request.form["Num_Credit_Inquiries"]
        Delay_from_due_date = request.form["Delay_from_due_date"]
        
        X = np.array([[float(Credit_Mix), float(Outstanding_Debt), float(Interest_Rate),
                       float(Payment_of_Min_Amount), float(Num_Credit_Inquiries), float(Delay_from_due_date)]])
        #category = str(rfmodel.predict(X)[0])
        category = "standard"
        pred = rfmodel.predict_proba(X).max()
    return render_template("index.html", pred=pred, category=category)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
