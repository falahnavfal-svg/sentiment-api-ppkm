import requests  # Library untuk mengirim permintaan HTTP
import json

# Alamat API kita
url = 'http://127.0.0.1:5000/predict'

# 1. Tes dengan kalimat NEGATIF
print("Menguji sentimen NEGATIF...")
data_negatif = {
    'teks': 'ppkm diperpanjang terus bikin susah pedagang kecil #tolakppkm'
}
try:
    response_negatif = requests.post(url, json=data_negatif)
    print("Respon dari server:")
    print(json.dumps(response_negatif.json(), indent=2, ensure_ascii=False))
except Exception as e:
    print(
        f"Error: Tidak bisa terhubung ke server. Pastikan app.py sudah jalan. ({e})")


print("\n" + "="*30 + "\n")

# 2. Tes dengan kalimat POSITIF
print("Menguji sentimen POSITIF...")
data_positif = {
    'teks': 'terima kasih bantuannya pak @jokowi, bansos sangat membantu kami'
}
try:
    response_positif = requests.post(url, json=data_positif)
    print("Respon dari server:")
    print(json.dumps(response_positif.json(), indent=2, ensure_ascii=False))
except Exception as e:
    print(
        f"Error: Tidak bisa terhubung ke server. Pastikan app.py sudah jalan. ({e})")
