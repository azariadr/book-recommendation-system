# Laporan Proyek Machine Learning - Azaria Dhea Rismaya

## Project Overview

Membaca buku adalah kegiatan penting yang memberikan banyak manfaat, seperti memperluas wawasan dan pengetahuan. Seiring dengan perkembangan zaman, jumlah buku yang diterbitkan semakin banyak, sehingga menyulitkan pembaca dalam menemukan buku yang sesuai dengan minat dan preferensi mereka. Sistem rekomendasi buku hadir sebagai solusi untuk mengatasi masalah ini.

Tujuan dari proyek ini adalah untuk membangun sistem rekomendasi buku yang dapat membantu pembaca dalam menemukan buku-buku yang relevan dengan minat mereka. Sistem ini akan menggunakan dua pendekatan, yaitu content-based filtering dan collaborative filtering.

## Business Understanding

### Problem Statements

- Bagaimana cara membangun sistem rekomendasi buku yang dapat memberikan saran buku berdasarkan riwayat bacaan pengguna sebelumnya?
- Bagaimana cara merekomendasikan buku dengan mempertimbangkan kesamaan penulis dengan buku yang telah dibaca pengguna?

### Goals

- Mengembangkan sistem rekomendasi buku yang dapat memberikan saran buku yang relevan dan personal kepada pengguna berdasarkan riwayat bacaan mereka.
- Mengetahui cara merekomendasikan buku dengan penulis yang sama atau serupa dengan buku yang telah dibaca pengguna.

### Solution Approach
Proyek ini menggunakan pendekatan Content Based Filtering dan Collaborative Based Filtering
- Pendekatan content-based filtering akan merekomendasikan buku-buku yang memiliki kesamaan dengan buku yang telah dibaca atau disukai oleh pengguna sebelumnya. Kesamaan ini dapat diukur berdasarkan berbagai atribut, seperti penulis, genre, atau kata kunci. Sistem akan merekomendasikan buku yang ditulis oleh penulis yang sama dengan buku yang pernah dibaca oleh user.
- Pendekatan collaborative filtering akan merekomendasikan buku-buku yang telah dibaca dan disukai oleh pengguna lain yang memiliki preferensi serupa. Sistem akan mengidentifikasi pengguna-pengguna lain yang memiliki riwayat membaca dan rating yang mirip dengan pengguna target, dan kemudian merekomendasikan buku-buku yang telah dibaca dan disukai oleh pengguna-pengguna tersebut.

## Data Understanding
Proyek ini menggunakan dataset yang berisi informasi tentang buku dan pengguna. Dataset ini akan digunakan untuk melatih dan menguji sistem rekomendasi. Dataset yang digunakan adalah Book-Recommendation-Dataset (https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset).

Berikut informasi pada dataset :
- Format CSV
- Terdiri dari 3 dataframe, yaitu Books.csv, Ratings.csv, dan Users.csv. Namun, pada proyek ini hanya digunakan 2 dataframe, yaitu Books.csv dan Ratings.csv.
- Books.csv berjumlah 271.360 sampel dengan 8 fitur.
- Ratings.csv berjumlah 1.149.780 sampek dengan 3 fitur.
- Lengkap (tidak ada missing value)

Variabel-variabel pada Books.csv adalah sebagai berikut:
- ISBN (String) :	Nomor identifikasi buku yang unik (International Standard Book Number)
- Book-Title (String) :	Judul buku
- Book-Author (String) :	Nama penulis buku
- Year-Of-Publication (String) :	Tahun penerbitan buku
- Publisher (String) :	Penerbit buku
- Image-URL-S (String) :	URL gambar sampul buku (ukuran kecil)
- Image-URL-M (String) :	URL gambar sampul buku (ukuran sedang)
- Image-URL-L (String) :	URL gambar sampul buku (ukuran besar)

Variabel-variabel pada Ratings.csv adalah sebagai berikut:
- User-ID (int) :	ID pengguna yang memberikan rating
- ISBN (String) :	Nomor identifikasi buku yang diberi rating
- Book-Rating (int) :	Rating yang diberikan oleh pengguna (skala 0-10)

### Exploratory Data Analysis (EDA)
Bar plot digunakan untuk menampilkan frekuensi atau jumlah data dalam setiap kategori. Dalam contoh ini, kita melihat distribusi data pada fitur 'year_of_publication' dan 'rating' untuk mengetahui sebaran rating buku dan berapa banyak buku yang dipublikasikan pada setiap tahun.

![Distribusi Tahun Publikasi Buku](https://github.com/user-attachments/assets/c196e0f9-f6fe-4573-a655-fe74568a276b)

![Distribusi Rating Buku](https://github.com/user-attachments/assets/5461e3a6-d64c-4d8c-8fa3-17f3b69688b5)

Pair plot menampilkan scatter plot untuk setiap pasangan fitur numerik dalam dataset. Ini membantu untuk mengidentifikasi korelasi dan pola hubungan antar fitur. diag_kind='kde' digunakan untuk menampilkan kernel density estimate (KDE) pada diagonal, yang memberikan informasi tentang distribusi data untuk setiap fitur.

![Pairplot](https://github.com/user-attachments/assets/3dcd3a84-699f-46ce-8c96-2f1437e2f3e6)

## Data Preparation
Nama kolom pada DataFrame `ratings` dan `books` diubah menggunakan rename agar lebih mudah diakses dan konsisten. Nama kolom `Book-Rating` diubah menjadi `rating` dan `User-ID` diubah menjadi `user_id` pada DataFrame `ratings`.Nama kolom `Book-Title` diubah menjadi `book_title`, `Book-Author` diubah menjadi `book_author`, `Year-Of-Publication` diubah menjadi `year_of_publication`, `Image-URL-S` diubah menjadi `Image_URL_S`, `Image-URL-M` diubah menjadi `Image_URL_M`, dan `Image-URL-L` diubah menjadi `Image_URL_L` pada DataFrame `books`. Proyek ini hanya menggunakan 10.000 data Books.csv dan 5.000 data Ratings.csv untuk mempercepat proses training model. Buku-buku dengan rating tertinggi disimpan dalam list `best_books`. Dilakukan penghapusan baris yang mengandung nilai missing (NaN) menggunakan dropna. Pembersihan data dilakukan dengan menghilangkan baris yang tidak lengkap, sehingga data yang digunakan untuk analisis lebih valid. Baris duplikat dihapus menggunakan drop duplicates untuk membersihkan data dengan memastikan setiap baris data unik, sehingga menghindari bias dalam analisis.

### Content Based Filtering
DataFrame `books` disederhanakan dengan hanya mengambil kolom-kolom yang relevan untuk tugas rekomendasi buku, yaitu ISBN, judul, penulis, dan tahun penerbitan. DataFrame `book` yang baru dibuat ini akan lebih mudah digunakan dalam proses selanjutnya, seperti perhitungan kemiripan antar buku. TF-IDF (Term Frequency-Inverse Document Frequency): untuk menghitung bobot kata dalam nama penulis buku. Bobot kata ini akan digunakan untuk mengukur kesamaan antar buku berdasarkan penulisnya. TfidfVectorizer digunakan untuk membuat representasi numerik dari penulis buku berdasarkan frekuensi kemunculan kata dalam nama penulis. Setiap penulis direpresentasikan sebagai vektor yang menunjukkan bobot kata dalam nama penulis. Objek `TfidfVectorizer` dibuat untuk menghitung nilai TF-IDF. `TfidfVectorizer` dilatih pada kolom `book_author` dari DataFrame `book`. Ini akan mempelajari kosakata dan menghitung IDF (Inverse Document Frequency) untuk setiap kata dalam nama penulis. Daftar fitur (kata-kata) yang telah dipelajari oleh TfidfVectorizer akan didapatkan menggunakan `tf.get_feature_names_out()`. Matriks TF-IDF untuk setiap buku dihitung berdasarkan nama penulisnya. Setiap baris dalam matriks mewakili sebuah buku, dan setiap kolom mewakili sebuah kata dalam kosakata. Nilai dalam matriks menunjukkan bobot TF-IDF untuk setiap kata dalam nama penulis buku tersebut. Dimensi matriks TF-IDF (jumlah buku dan jumlah fitur) akan ditampilkan melalui `tfidf_matrix.shape`. Matriks TF-IDF yang sparse dikonversi menjadi matriks dense. DataFrame dari matriks TF-IDF yang dense dibuat dengan kolom sebagai fitur (kata-kata) dan baris sebagai judul buku. Kemudian, mengambil sampel acak 10 kolom dan 10 baris untuk ditampilkan.

### Collaborative Based Filtering
Variabel `user_id` dan `ISBN` (yang mungkin berupa string atau angka yang tidak berurutan) diubah menjadi representasi numerik (integer) yang berurutan mulai dari 0. Lalu, dilakukan pengacakan kolom `rating` agar model tidak belajar pola yang tidak diinginkan dari urutan data asli. Selanjutnya, tipe data kolom `rating` diubah menjadi float32 untuk kompatibilitas dengan TensorFlow dan dinormalisasi ke rentang 0 hingga 1 menggunakan Min-Max scaling. Data dibagi menjadi data training (70%) dan data validasi (30%). Data training digunakan untuk melatih model, sedangkan data validasi digunakan untuk mengukur performa model pada data yang belum pernah dilihat sebelumnya.

## Modeling
### Content Based Filtering
Proyek ini membangun sistem rekomendasi buku berbasis konten yang merekomendasikan buku-buku dengan penulis yang serupa dengan buku yang telah dibaca pengguna. Pendekatan yang digunakan adalah Cosine Similarity untuk mengukur kesamaan antar buku berdasarkan bobot kata dalam nama penulis yang dihitung menggunakan TF-IDF. `cosine_similarity` dari library scikit-learn digunakan untuk menghitung kemiripan antar buku berdasarkan representasi vektor TF-IDF mereka. Cosine similarity mengukur sudut antara dua vektor, dengan nilai mendekati 1 menunjukkan kemiripan yang tinggi dan nilai mendekati 0 menunjukkan kemiripan yang rendah. Hasil perhitungan disimpan dalam matriks cosine similarity di mana setiap elemen (i, j) menunjukkan kemiripan antara buku i dan buku j.

Fungsi `author_recommendations` menerima judul buku yang telah dibaca sebagai input. Fungsi mencari buku input dalam matriks cosine similarity. Fungsi mengurutkan buku-buku lain berdasarkan nilai cosine similarity terhadap buku input (dari yang paling mirip hingga yang paling tidak mirip). Fungsi mengembalikan daftar buku yang paling mirip (direkomendasikan) selain buku input.

<img width="328" alt="Screenshot 2025-01-17 175127" src="https://github.com/user-attachments/assets/0a39b71f-34fb-4503-b871-dc6f0d256af9" />

### Collaborative Based Filtering
Model RecommenderNet didefinisikan sebagai subclass dari tf.keras.Model. Model menggunakan layer embedding untuk user dan buku, serta layer bias. Fungsi aktivasi sigmoid digunakan untuk menghasilkan prediksi rating dalam rentang 0 hingga 1. Model dikompilasi dengan fungsi loss BinaryCrossentropy, optimizer Adam, dan metrik RootMeanSquaredError. Model dilatih menggunakan data training dengan model.fit().

<img width="430" alt="Screenshot 2025-01-17 175429" src="https://github.com/user-attachments/assets/3db98db0-c9b0-42be-bed2-30443feedc44" />

## Evaluation
### Content Based Filtering
Akurasi model rekomendasi dievaluasi dengan menghitung persentase buku yang direkomendasikan yang memiliki penulis yang sama dengan buku yang telah dibaca oleh pengguna.

<img width="149" alt="Screenshot 2025-01-17 175650" src="https://github.com/user-attachments/assets/ed5c68cc-d142-4c4e-84e6-97db0361951b" />

### Collaborative Based Filtering
Model dievaluasi menggunakan metrik Root Mean Squared Error (RMSE). 

RMSE = √(Σ(yᵢ - ŷᵢ)² / n)
Keterangan:
- yᵢ: Nilai aktual (observasi) ke-i.
- ŷᵢ: Nilai prediksi ke-i.
- n: Jumlah total data.
- Σ: Simbol sigma, yang berarti penjumlahan dari semua nilai.

RMSE mengukur rata-rata perbedaan kuadrat antara prediksi rating dan rating sebenarnya. Semakin rendah nilai RMSE, semakin baik performa model. Model berhasil dilatih dan menunjukkan penurunan nilai RMSE selama proses training.

![download](https://github.com/user-attachments/assets/c01b89dc-f2d0-4741-b273-af48ca98e0cb)
