import joblib  # Untuk memuat model
import re      # Untuk membersihkan teks (harus sama persis)
from flask import Flask, request, jsonify

app = Flask(__name__)
try:
    model = joblib.load('sentiment_model_ppkm.joblib')
except FileNotFoundError:
    print("Error: File model 'sentiment_model_ppkm.joblib' tidak ditemukan!")
    exit()


def bersihkan_teks(teks):
    teks = str(teks)
    teks = re.sub(r'@[A-Za-z0-9_]+', '', teks)
    teks = re.sub(r'#\w+', '', teks)
    teks = re.sub(r'https://\S+', '', teks)
    teks = re.sub(r'http\S+', '', teks)
    teks = re.sub(r'[^A-Za-z ]+', '', teks)
    teks = teks.lower().strip()
    teks = re.sub(r'\s+', ' ', teks)
    return teks


@app.route('/predict', methods=['POST'])
def predict_sentiment():
    try:
        data = request.get_json()
        teks_input = data['teks']
        teks_bersih = bersihkan_teks(teks_input)
        prediksi = model.predict([teks_bersih])
        hasil_sentimen = prediksi[0]
        return jsonify({
            'teks_input': teks_input,
            'teks_bersih': teks_bersih,
            'sentimen': hasil_sentimen
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(port=5000, debug=True)
