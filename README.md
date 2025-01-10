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

## **_Exploratory Data Analysis_**

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
```python  <<class 'pandas.core.frame.DataFrame'>
RangeIndex: 12294 entries, 0 to 12293
Data columns (total 7 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   anime_id  12294 non-null  int64  
 1   name      12294 non-null  object 
 2   genre     12232 non-null  object 
 3   type      12269 non-null  object 
 4   episodes  12294 non-null  object 
 5   rating    12064 non-null  float64
 6   members   12294 non-null  int64  
dtypes: float64(1), int64(2), object(4)
memory usage: 672.5+ KB
 ```
Masih ada beberapa tindakan yang perlu dilakukan untuk `anime_df`. Proses pembersihan dan persiapan dataset akan dikerjakan lebih lanjut pada tahap selanjutnya.
Berikut ini adalah EDA yang dilakukan untuk `rating_df`:
```python
  rating_df.shape
  ```
  Kode diatas memiliki output:
  ```python
  (93045, 3)
  ```
  Berdasarkan output diatas, `movie_df` memiliki:
  - 93045 baris data
  - 3 kolom data
  
 ```python
rating_df.keys()
 ```
  Kode diatas memiliki output:
 ```python
 Index(['user_id', 'anime_id', 'rating'], dtype='object')
 ```
    
```python
rating_df.info()
```
Kode diatas memiliki output:
```python <class 'pandas.core.frame.DataFrame'>
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
```python
rating_df['rating'].describe()
```
Kode diatas memiliki output:
```python
 	rating
count 	93045.000000
mean 	6.114826
std 	3.778899
min 	-1.000000
25% 	6.000000
50% 	8.000000
75% 	9.000000
max 	10.000000

dtype: float64
```
##  **_Data Vizualisasi_**
Visualisasi Data untuk `anime_df`:

* Univariate Analysis
  
  ![Untitled](https://github.com/user-attachments/assets/5b651d6e-7763-4b76-9c64-b7502ca4cfe5)
  Gambar 1. Anime Categories Distribution

Berdasarkan visualisasi diatas distribusi anime terbesar di media TV

  ![Untitled](https://github.com/user-attachments/assets/50441481-d7af-4b03-99a9-7d5bd0d79128)
  Gambar 2. Anime's Average Ratings Distribution

Visualisasi Data untuk `rating_df`:

 ![Untitled](https://github.com/user-attachments/assets/f812ea7f-d82e-41d8-8969-caeb8f81d028)
 Gambar 3. Count Plot Rating

 ![Untitled](https://github.com/user-attachments/assets/432ed6e3-c5d5-48dc-827c-5fa925a33f2a)
 Gambar 4.  Pie Chart Rating

* Multivariate Analysis
 ```python
anime_df.sort_values(by='members', ascending=False).head(10)
```
| 	|anime_id| 	name |	genre| 	type| 	episodes |	rating 	|members|
|---|-------|-------|------|-------|--------|----------|--------|
|40 |1535 |	Death Note |	Mystery, Police, Psychological, Supernatural, ...| 	TV |	37 |	8.71| 	1013917|
|86| 	16498 |	Shingeki no Kyojin| 	Action, Drama, Fantasy, Shounen, Super Power| 	TV| 	25|	8.54| 	896229|
|804|	11757 |	Sword Art Online |	Action, Adventure, Fantasy, Game, Romance| 	TV |	25 |	7.83| 	893100|
|1 |	5114 	|Fullmetal Alchemist: Brotherhood 	|Action, Adventure, Drama, Fantasy, Magic, Mili...| 	TV| 	64 |	9.26 |	793665|
|159| 	6547| 	Angel Beats! 	|Action, Comedy, Drama, School, Supernatural| 	TV| 	13 |	8.39 |	717796|
|19 |	1575 |	Code Geass: Hangyaku no Lelouch |	Action, Mecha, Military, School, Sci-Fi, Super...| 	TV |	25 	|8.83| 	715151|
|841| 	20 |	Naruto 	|Action, Comedy, Martial Arts, Shounen, Super P...| 	TV 	|220| 	7.81| 	683297|
|3 |	9253 |	Steins;Gate 	|Sci-Fi, Thriller| 	TV 	|24 	|9.17 |	673572|
|445| 	10620| 	Mirai Nikki (TV) 	|Action, Mystery, Psychological, Shounen, Super...| 	TV |	26 |	8.07| 	657190|
|131| 	4224 |	Toradora! 	|Comedy, Romance, School, Slice of Life| 	TV 	|25 |	8.45 	|633817|

Menampilkan daftar anime dengan jumlah anggota community terbanyak. Misalnya, anime Death Note memiliki jumlah anggota community terbanyak, yaitu sebesar 1013917.

## Missing value

Missing Values adalah data yang hilang atau tidak tercatat dalam dataset. Hal ini bisa terjadi karena berbagai alasan, seperti kesalahan entri data, kerusakan data, atau tidak tersedianya informasi saat pengumpulan data. Missing values dapat mempengaruhi kualitas model machine learning dan hasil analisis statistik. Oleh karena itu, penting untuk mengidentifikasi, menganalisis, dan mengatasi missing values dengan metode seperti imputasi, di mana nilai yang hilang diganti dengan estimasi, atau dengan menghapus baris atau kolom yang terdampak.

Dataset untuk `anime_df`:
```python
anime_df.isnull().sum()
```
```python
 	       0
anime_id 	0
name 	0
genre 	62
type 	25
episodes 	0
rating 	230
members 	0

dtype: int64
```
Berdasarkan hasil diatas, terdapat 3 missing values yaitu genre, type, rating

Dataset untuk `rating_df`:
```python
rating_df.isnull().sum()
```
```python
 	      0
user_id 	0
anime_id 	0
rating 	0

dtype: int64
```
Tidak ada missing value

## Duplikat data

```python
# Cek baris duplikat dalam dataset
duplicates_rating = rating_df.duplicated()

# Hitung jumlah baris duplikat
duplicate_rating= duplicates_rating.sum()

# Cetak jumlah baris duplikat
print(f"Number of duplicate rows: {duplicate_rating}")
```
```python
Number of duplicate rows: 0
```

```python
# Cek baris duplikat dalam dataset
duplicates_anime = anime_df.duplicated()

# Hitung jumlah baris duplikat
duplicate_anime = duplicates_anime.sum()

# Cetak jumlah baris duplikat
print(f"Number of duplicate rows: {duplicate_anime}")
```
```python
Number of duplicate rows: 0
```
Berdasarkan hasil tersebut, tidak ditemukan adanya data duplikat, maka tidak ada juga proses penghapusannya.

## Missing values
```python
rating_df['rating'].describe()
```
```python
 	    rating
count 	93045.000000
mean 	6.114826
std 	3.778899
min 	-1.000000
25% 	6.000000
50% 	8.000000
75% 	9.000000
max 	10.000000
```
Dataset rating anime memiliki rating terendah yang diberikan user pada suatu anime adalah -1 dan rating tertinggi adalah 10. Rating -1 menandakan bahwa user menonton anime, namun tidak memberikan rating.

# Data Preparation
## Data Cleaning
### Removal Duplicates

**1 duplikat data akan kita hapus dalam rating_df**

```python
anime_df=anime_df.drop_duplicates()
```
Berhasil dihapus duplikat data

### Handle Missing Value

Berdasarkan hasil diatas, terdapat 3 missing values yaitu `genre`` type`` rating`.

```python
anime_df.dropna(inplace =True)
anime_df.shape
```
Missing value Berhasil ditangani.

### Outliers Detection and Removal

Berdasarkan output data understanding, terlihat bahwa nilai terkecil dari review adalah -1 dan terbesarnya adalah 10. Rating -1 menandakan bahwa user menonton anime, namun tidak memberikan rating.
```python
rating_df = rating_df[~(rating_df.rating == -1)]
rating_df.describe().apply(lambda s: s.apply('{0:.2f}'.format))
```
```python
|      |user_id| anime_id|rating|
|-------|--------|--------|------|
|count |	74776.00 |	74776.00 |74776.00|
|mean |	498.80 |	10640.70 |	7.85|
|std 	|267.17 	|9016.21 |	1.54|
|min 	|1.00 	|1.00 	|1.00|
|25% |	277.00 |	2236.00| 	7.00|
|50% |	508.50 |	9367.00 |	8.00|
|75% 	|734.00 |	16512.00| 	9.00|
|max 	|958.00 |34240.00 	|10.00|

Sebelum memulai dengan proses interquartile. Perlu dilihat terlebih dahulu secara sekilas secara statistika deskriptif.
Berdasarkan output diatas, terlihbat bahwa nilai terkecil dari score adalah 1 dan terbesarnya adalah 10.

### Menghapus symbol pada judul anime

```python
import re
def text_cleaning(text):
    text = re.sub(r'"', '', text)
    text = re.sub(r'.hack//', '', text)
    text = re.sub(r'"', '', text)
    text = re.sub(r'A"s', '', text)
    text = re.sub(r'I"', 'I\'', text)
    text = re.sub(r'&', 'and', text)

    return text

anime_df['name'] = anime_df['name'].apply(text_cleaning)
```
Symbol dalam judul sudah dihapus

```python
```
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
