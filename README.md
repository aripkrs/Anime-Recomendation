# Laporan Proyek Anime Recomendation-Arip Kristiyanto
# Project Overview
Anime adalah salah satu media hiburan yang saat ini sedang banyak diperbincangkan. Anime
sendiri merupakan film animasi dengan teknik penggambaran yang melibatkan emosi dari setiap
karakter dengan alur yang kompleks[1]. Hal tersebut membuat anime menjadi salah satu media
hiburan yang banyak digemari oleh masyarakat di Indonesia. Namun, dikarenakan banyaknya judul
dan genre anime yang berbeda, membuat banyak penggemar anime kesusahan untuk mencari judul
atau genre anime yang sesuai dengan yang ingin mereka tonton. Oleh karena itu, sistem rekomendasi
anime menjadi peran penting.

Sistem rekomendasi adalah suatu program yang melakukan prediksi sesuatu item, seperti
rekomendasi film, musik, buku, berita dan lain sebagainya yang menarik user[2]. Penggunaan sistem
rekomendasi dalam dunia digital semakin berkembang pesat. Sistem rekomendasi dapat membantu
pengguna dalam memperoleh informasi yang relevan dengan preferensi mereka.

Ada beberapa pendekatan yang dapat digunakan dalam membuat model sistem rekomendasi film. 
Pendekatan berbasis konten memanfaatkan fitur-fitur seperti genre untuk merekomendasikan film serupa [3]. 
Sementara itu, pendekatan pemfilteran kolaboratif mengumpulkan dan menganalisis data dari banyak pengguna untuk menemukan pola dan preferensi bersama[4]. 
Perkembangan model ini terus berkembang seiring dengan kemajuan teknologi dan analisis data.
# Business Understanding
Pengguna saat ini dimana konsumsi konten melalui platform streaming semakin meningkat,
kebutuhan akan sistem rekomendasi yang akurat menjadi semakin penting, khususnya di industri
animasi. Hal ini berfokus pada penerapan sistem rekomendasi yang dapat membantu pemirsa
dengan mudah menavigasi banyaknya konten.

## Problem Statements

 - Bagaimana memahami dan mengetahui terkait data dari dataset digunakan untuk pembuatan model sistem rekomendasi?
 - Bagaimana membuat model sistem rekomendasi dengan pendekatan content-based filtering?
 - Bagaimana membuat model sistem rekomendasi dengan pendekatan collaborative filtering?
 - Bagaimana cara mengukur performa model sistem rekomendasi yang sudah dibuat?

## Goals
- Melakukan langkah-langkah untuk memahami dataset terlebih dahulu, seperti EDA dan Visualisasi Data.
- Membuat sistem rekomendasi anime dengan content-based filtering.
- Membuat sistem rekomendasi anime dengan collaborative filtering.
- Melakukan evaluasi terhadap model sistem rekomendasi yang telat dibuat.

## Solution Approach
- Melakukan EDA untuk mengeksplorasi fitur menggunakan fungsi `shape`, `key`, `info` pada dataset. Kemudian, dilakukan visualisasi data seperti count plot dan pie chart untuk mendapatkan gambaran atau ilustrasu lebih jelas mengenai dataset yang digunakan.
- Membangun sistem rekomendasi dengan `content-based filtering` yang memberikan rekomendasi kepada pengguna berdasarkan kesamaan pada item yang ada. Data yang digunakan berisi data dari genre dari setiap film. Dataset tersebut juga melewati tahap Data Preparation agar dataset dapat digunakan untuk proses pembangunan model seperti, menangani data duplikat, missing value, dan mengganti beberapa data agar sesuai. Kemudian, data yang sudah siap, diproses ke tahap modelling yang memanfaatkan `Tfidvectorizer`, `cosine similarity`, dan fungsi buatan yang mengembalikan rekomendasi berdasarkan kesamaan pada item. Pendekatan ini berfokus pada karakteristik atau konten dari item yang direkomendasikan
- Membangun sistem rekomendasi dengan `collaborative filtering` yang memberikan rekomendasi kepada pengguna dengan menganalisis perilaku dan preferensi pengguna. Data yang digunakan berisi data review untuk film-film dari user. Dataset tersebut juga melewati tahap Data Preparation agar dataset dapat digunakan untuk proses pembangunan model seperti, menangani data duplikat, missing value, encoding, dan train test split. Kemudian, data yang sudah siap, diproses ke tahap modelling yang menggunakan `RecommenderNet` dan `Early Stopper` dalam proses training-nya. Pendekatan ini membutuhkan data terkait user. 
- Melakukan perhitungan skor presisi untuk mengukur performa dari model sistem rekomendasi film dengan content-based learning. Kemudian, menggunakan skor RMSE atau root mean squared error untuk mengukur performa dari model sistem rekomendasi film dengan colaborative filtering.

# Data Understanding
Dataset yang digunakan untuk pembuatan model system recommendation ini adalah dataset "📺 Anime ⛩️ Recomendation Systems 🔺🔻" yang tersedia di situs [kaggle](https://www.kaggle.com/code/dumanmesut/anime-recomendation-systems) yang berisi data-data mengenai anime beserta rating yang diberikan oleh para penggemar.

Terdapat 2 file didalamnya, dataset animes.csv dan rating.csv. animes terdiri dari , baris 12294 data dan 7 kolom data. rating.csv terdiri dari 93045 baris data dan 3 kolom data.

Kedua dataset tersebut dapat digunakan untuk membuat system recommendation, baik Content-Based Filtering maupun Collaborative Filtering

Berikut ini adalah infomasi lainnya mengenai atribut-atribut yang terdapat pada dua dataset tersebut:

Atribut-atribut pada anime_df.csv:

   - ```anime_id``` - identifikasi unik anime
   - ```name``` - full name .
   - ```genre``` - genre dari anime
   - ```type``` - movie, TV, OVA, etc.
   - ```episodes``` - berapa banyak episode dalam acara ini. (1 jika film)
   - ```rating``` - peringkat rata-rata dari 10 untuk anime
   - ```members``` - umlah anggota komunitas yang ada di grup anime

Atribut-atribut pada rating_df.csv:

    - ```user_id``` - ID pengguna yang dibuat secara acak.
    - ```anime_id``` - anime yang diberi peringkat oleh pengguna ini.
    - ```rating``` - Peringkat dari 10 yang diberikan pengguna ini (-1 jika pengguna menontonnya tetapi tidak memberikan peringkat).

**_Exploratory Data Analysis_**

Exploratory Data Analysis (EDA) adalah pendekatan analisis data yang bertujuan untuk memahami karakteristik utama dari kumpulan data. EDA melibatkan penggunaan teknik statistik dan visualisasi grafis untuk menemukan pola, hubungan, atau anomali untuk membentuk hipotesis. Proses ini sering kali tidak terstruktur dan dianggap sebagai langkah awal penting dalam analisis data yang membantu menentukan arah analisis lebih lanjut.

Berikut ini adalah EDA yang dilakukan untuk `anime_df`:
```python
  anime_df.shape
  ```
  Kode diatas memiliki output:
  ```python
  (12294, 7)
  ```
  Berdasarkan output diatas, `movie_df` memiliki:
  - 12294 baris data
  - 7 kolom data
  
 ```python
anime_df.keys()
 ```
  Kode diatas memiliki output:
 ```python
 Index(['anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'members'], dtype='object')
 ```
    
```python
anime_df.info()
```
```python  <class 'pandas.core.frame.DataFrame'>
RangeIndex: 93045 entries, 0 to 93044
Data columns (total 3 columns):
 #   Column    Non-Null Count  Dtype
---  ------    --------------  -----
 0   user_id   93045 non-null  int64
 1   anime_id  93045 non-null  int64
 2   rating    93045 non-null  int64
dtypes: int64(3)
memory usage: 2.1 MB
 ```

# Data Preparation

# Modeling

# Evaluation
# Referensi

**[1] D. Domarco and N. M. S. Iswari, “Rancang Bangun Aplikasi Chatbot Sebagai Media
Pencarian Informasi Anime Menggunakan Regular Expression Pattern Matching,” J. Ultim.,
vol. 9, no. 1, pp. 19–24, 2017, doi: 10.31937/ti.v9i1.559.**

**[2] F. W. M. Fadlil, “Pembuatan Aplikasi Rekomendasi Menggunakan Decision Tree dan
Clustering,” vol. 3, no. Kursor, pp. 45–46, 2007.**

**[3] “Movie Recommendation Systems Using Content-Based Filtering,” International Research Journal of Modernization in Engineering Technology and Science, 
Jun. 2023, doi: https://doi.org/10.56726/irjmets42626.**

**[4] S. Katkam, A. Atikam, P. Mahesh, M. Chatre, S. S. Kumar, and S. G. R, “Content-based Movie Recommendation System and Sentimental analysis using ML,”
IEEE Xplore, May 01, 2023. https://ieeexplore.ieee.org/document/10142424**
