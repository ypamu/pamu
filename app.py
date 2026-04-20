import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Memuat model DAN transformer yang baru saja kamu simpan
model = pickle.load(open("linear_regression_model.pkl", "rb"))
transformer = pickle.load(open("transformer.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # 1. Ambil data dari form
    data = {
        'Age': [float(request.form['Age'])],
        'Gender': [request.form['Gender']],
        'Blood Type': [request.form['Blood Type']],
        'Medical Condition': [request.form['Medical Condition']]
    }
    
    # 2. Ubah ke DataFrame agar bisa diproses transformer
    input_df = pd.DataFrame(data)
    
    # 3. Transformasi data kategori menjadi angka (One-Hot Encoding)
    transformed_data = transformer.transform(input_df)
    
    # 4. Prediksi
    prediction = model.predict(transformed_data)
    output = round(prediction[0], 2)

    return render_template(
        "index.html", 
        prediction_text="Estimasi Tagihan Medis: ${}".format(output)
    )

if __name__ == "__main__":
    app.run(debug=True)
