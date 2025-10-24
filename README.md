API Klasifikasi Sentimen PPKM (End-to-End Project)

Ini adalah proyek portofolio end-to-end Machine Learning yang membuat sebuah API untuk menganalisis sentimen (Positif/Negatif) dari teks berbahasa Indonesia, khususnya terkait topik PPKM.
Proyek ini mencakup seluruh alur kerja Data Science, mulai dari pembersihan data mentah, pelatihan model, hingga deployment model sebagai API yang live di cloud.
URL API yang Live: https://huggingface.co/spaces/falahnfl/api-sentimen-ppkm

1. Tech Stack
Proyek ini dibangun menggunakan:
Python: Bahasa pemrograman utama.
Scikit-learn (sklearn): Untuk melatih model klasifikasi (LinearSVC) dan membuat pipeline (TfidfVectorizer).
Pandas: Untuk memuat dan membersihkan dataset.
Flask: Untuk membuat endpoint API RESTful.
Gunicorn: Sebagai server WSGI untuk production.
Docker: Untuk membuat container aplikasi.
Hugging Face Spaces: Sebagai platform cloud (PaaS) untuk men-deploy API.

2. Proses Proyek
Data: Model ini dilatih menggunakan dataset INA_TweetsPPKM_Labeled_Pure.csv, yang berisi ~21.000 tweet berlabel positif (1) dan negatif (2).
Training: Model (LinearSVC dengan class_weight='balanced') dilatih untuk menangani data yang tidak seimbang dan disimpan sebagai file sentiment_model_ppkm.joblib.
API: Sebuah API Flask sederhana dibuat di app.py yang memuat model .joblib dan menyediakan endpoint /predict.
Deployment: Aplikasi ini di-container-kan menggunakan Dockerfile dan di-deploy ke Hugging Face Spaces.

3. Cara Menggunakan API (Contoh Python)
API ini menerima request POST dengan JSON payload yang berisi key "teks".
Anda bisa mengujinya dengan skrip Python test_api.py
