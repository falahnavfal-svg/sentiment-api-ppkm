import pandas as pd
import re
import joblib  # Untuk menyimpan model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


def bersihkan_teks(teks):
    teks = str(teks)  # Pastikan input adalah string
    teks = re.sub(r'@[A-Za-z0-9_]+', '', teks)  # Hapus @username
    teks = re.sub(r'#\w+', '', teks)  # Hapus #hashtag
    teks = re.sub(r'https://\S+', '', teks)  # Hapus link https
    teks = re.sub(r'http\S+', '', teks)  # Hapus link http
    # Hapus karakter non-alfabet (angka dan tanda baca)
    teks = re.sub(r'[^A-Za-z ]+', '', teks)
    teks = teks.lower().strip()  # Ubah jadi huruf kecil dan hapus spasi berlebih
    teks = re.sub(r'\s+', ' ', teks)  # Ganti spasi ganda/dll dengan satu spasi
    return teks


file_path = 'INA_TweetsPPKM_Labeled_Pure.csv'
try:
    data = pd.read_csv(file_path, delimiter='\t',
                       usecols=['Tweet', 'sentiment'])
except FileNotFoundError:
    print(f"Error: File {file_path} tidak ditemukan.")
    print("Pastikan file CSV ada di folder yang sama dengan skrip Python ini.")
    exit()
data_filtered = data[data['sentiment'].isin([1, 2])].copy()

label_map = {
    1: 'positif',
    2: 'negatif'
}

data_filtered['sentiment_label'] = data_filtered['sentiment'].map(label_map)
data_filtered['Tweet_bersih'] = data_filtered['Tweet'].apply(bersihkan_teks)

X = data_filtered['Tweet_bersih']
y = data_filtered['sentiment_label']

X = X[X.notna() & (X != '')]
y = y[X.index]  # Jaga agar label tetap sinkron
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"{len(X_train)}")
print(f"{len(X_test)}")

text_clf_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', LinearSVC(class_weight='balanced', max_iter=2000))
])
text_clf_pipeline.fit(X_train, y_train)

y_pred = text_clf_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi Model: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred,
      target_names=['negatif', 'positif']))

model_filename = 'sentiment_model_ppkm.joblib'
joblib.dump(text_clf_pipeline, model_filename)
print(f"\nModel telah disimpan ke file: {model_filename}")
