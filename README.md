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
Berdasarkan hasil tersebut, tidak ditemukan adanya data duplikat.

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

Data duplikat adalah baris data yang sama persis untuk setiap variabel yang ada. Dataset yang digunakan perlu diperiksa juga apakah dataset memiliki data yang sama atau data duplikat. Jika ada, maka data tersebut harus ditangani dengan menghapus data duplikat tersebut.

Alasan: Data duplikat perlu didektesi dan dihapus karena jika dibiarkan pada dataset dapat membuat model Anda memiliki bias, sehingga menyebabkan overfitting. Dengan kata lain, model memiliki performa akurasi yang baik pada data pelatihan, tetapi buruk pada data baru. Menghapus data duplikat dapat membantu memastikan bahwa model Anda dapat menemukan pola yang ada lebih baik lagi.

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

Outliers adalah titik data yang secara signifikan berbeda dari sebagian besar data dalam kumpulan data. Outliers dapat muncul karena variasi dalam pengukuran atau mungkin menunjukkan kesalahan eksperimental; dalam beberapa kasus, outliers bisa juga menunjukkan variabilitas yang sebenarnya dalam data. Penting untuk menganalisis outliers karena mereka dapat memiliki pengaruh besar pada hasil analisis statistik.

Alasan:Outliers perlu dideteksi dan dihapus karena jika dibiarkan dapat merusak hasil analisis statistik pada kumpulan data sehingga menghasilkan performa model yang kurang baik. Selain itu, Mendeteksi dan menghapus outlier dapat membantu meningkatkan performa model Machine Learning menjadi lebih baik.

Berdasarkan output data understanding, terlihat bahwa nilai terkecil dari review adalah -1 dan terbesarnya adalah 10. Rating -1 menandakan bahwa user menonton anime, namun tidak memberikan rating.
```python
rating_df = rating_df[~(rating_df.rating == -1)]
rating_df.describe().apply(lambda s: s.apply('{0:.2f}'.format))
```

|    |user_id| anime_id|rating|
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
Symbol dalam judul berhasil dihapus

### Encoding
Encoding adalah proses konversi informasi dari satu bentuk atau format ke bentuk lain, yang sering kali dilakukan untuk memastikan kompatibilitas dan pemrosesan yang tepat oleh berbagai sistem komputer. Proses ini sangat penting dalam dunia digital, di mana berbagai jenis data, seperti teks, gambar, dan suara, harus diubah menjadi format yang dapat dipahami oleh perangkat keras dan perangkat lunak.

**Alasan**: Tahap ini perlu dilakukan karena Encoding memungkinkan data dari berbagai sumber dan format untuk diubah menjadi format standar yang dapat dipahami dan memastikan bahwa informasi dapat diproses

Berikut ini adalah proses dari encoding yang dilakukan:

```python
user_id = rating_df['user_id'].unique().tolist() # Mengubah userId menjadi list tanpa nilai yang sama
user_to_user = {x: i for i, x in enumerate(user_id)} # Melakukan encoding userId
user_encode_to_user = {i: x for i, x in enumerate(user_id)} # Melakukan proses encoding angka ke ke userId

print('list userId :  ', user_id)
print('encoded userId :  ', user_to_user)
print('encoded angka ke userId :  ', user_encode_to_user)
```
Berdasarkan kode diatas, proses encoding untuk userId.

```python
anime_id = rating_df['anime_id'].unique().tolist() # Mengubah movieId menjadi list tanpa nilai yang sama
anime_to_anime = {x: i for i, x in enumerate(anime_id)} # Melakukan proses encoding movieId
anime_encode_to_anime = {i: x for i, x in enumerate(anime_id)} # Melakukan proses encoding angka ke movieId

print('list anime_id:  ', anime_id)
print('encoded anime_id:  ', anime_to_anime)
print('encoded angka ke anime_id:  ', anime_encode_to_anime)
```
Berdasarkan kode diatas, proses encoding untuk anime_id.

```python
rating_df['user'] = rating_df['user_id'].map(user_to_user) # Mapping userId ke dataframe user
rating_df['anime'] = rating_df['anime_id'].map(anime_to_anime) # Mapping animeId ke dataframe resto
```
Hasil encoding tadi, di-mapping ke dalam dataframe review_df dengan menempati kolom baru untuk masing-masing hasil.

```python
rating_df.head(5)
```
|     |user_id |	anime_id |	rating| 	user 	|anime|
|------|-------|-------|---------|---------|------|
|47 	|1 	|8074| 	10| 	0| 	0|
|81 	|1 	|11617| 	10 	|0 |	1|
|83 	|1 	|11757 |	10 |	0| 	2|
|101 |1 	|15451 |	10| 	0| 	3|
|153 	|2 	|11771| 	10| 	1| 	4|

Proses mapping berhasil dilakukan karena sudah terdapat dua kolom baru, yaitu user dan anime

```python
num_users = len(user_to_user) # Mendapatkan jumlah user
num_anime = len(anime_to_anime) # Mendapatkan jumlah rating
min_rating = min(rating_df['rating']) # Nilai minimum rating
max_rating = max(rating_df['rating']) # Nilai maksimal rating

print('total user: {}'.format(num_users))
print('total rating: {}'.format(num_anime))
print('MIN rating {}'.format(min_rating))
print('MAX rating: {}'.format(max_rating))
```
Berdasarkan output diatas, dapat dilihat bahwa pada rating_df terdapat:
   * total user: 901
   * total rating: 4461
   * MIN rating: 1
   * MAX rating: 10

### Train Test Split

Train Test Split adalah metode yang digunakan untuk membagi dataset menjadi dua bagian: satu untuk melatih model (training set) dan satu lagi untuk menguji model (testing set). Biasanya, data dibagi dengan proporsi tertentu, misalnya 80% untuk training dan 20% untuk testing.

**Alasan**: Proses ini dilakukan agar dapat mengevaluasi kinerja model secara objektif. Dengan memisahkan data uji, kita dapat mengukur seberapa baik model memprediksi data baru yang tidak pernah dilihat sebelumnya, yang merupakan indikator penting dari kemampuan generalisasi model.

Berikut ini adalah proses Train Test Split yang dilakukan:
```python
rating_df = rating_df.sample(frac=1, random_state=18)     
rating_df
```
|  |user_id |	anime_id 	|rating| 	user| 	anime|
|------|-------|-------|---------|---------|------|
|36760 |	401 |	4181 |	10 	|376| 	557|
|85079 |	890 |	8115 |	7 |	837 |	3160|
|60255| 	627 |	8536 |	6 |	587 |	818|
|41755 |	446 |	29758| 	6 |	419| 	1349|
|56086 |	578 |	11887 |	8 |	540 |	831|  

Berdasarkan output diatas, proses shuffling atau pengacakan berhasil dilakukan

```python
x_df = rating_df[['user', 'anime']].values # Membuat variabel x_df untuk mencocokkan data user dan anime menjadi satu value
y_df = rating_df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values # Membuat variabel y_df untuk
```
Pemisahan rating_df menjadi dua bagian ke x_df dan y_df untuk proses Train Test Split berhasil dilakukan.

```python
# Membagi menjadi 90% data train dan 10% data validasi
train_indices = int(0.9 * rating_df.shape[0])
x_train, x_val, y_train, y_val = (
    x_df[:train_indices],
    x_df[train_indices:],
    y_df[:train_indices],
    y_df[train_indices:]
)
```
Proses Train Test Split telah dilakukan ke empat variabel berbebeda dengan komposisi 0.9 untuk train dan 0.1 untuk val. Berikut adalah keempatnya:

   * x_train
   * x_val
   * y_train
   * y_val
Proses Train Test Split berhasil dilakukan.

### TF-IDF (Term Frequency-Inverse Document Frequency)

Metode Term Frequency-Inverse Document Frequency (TF-IDF) adalah salah satu teknik yang digunakan dalam pengolahan teks dan pemodelan bahasa alami. Tujuan utama dari metode TF-IDF adalah untuk mengevaluasi seberapa penting suatu kata (term) dalam sebuah dokumen dalam konteks koleksi dokumen yang lebih besar.

Dalam metode TF-IDF, nilai TF dan IDF dikalikan bersama-sama untuk menghasilkan bobot kata (term weight) untuk setiap kata dalam sebuah dokumen. Bobot ini mencerminkan tingkat pentingnya kata dalam dokumen tersebut dibandingkan dengan koleksi dokumen yang lebih besar

TF-IDF digunakan pada sistem rekomendasi anime untuk menentukan representasi fitur penting dari setiap genre anime. Untuk menjalankan TF-IDF digunakan fungsi tfidfvectorizer() dari library sklearn.

Setelah itu hasil TF-IDF tadi ditransformasikan ke dalam bentuk matriks dengan fungsi todense().

**Inisialisasi TfidfVectorizer**
 
```python
tf_id = TfidfVectorizer()
tf_id.fit(anime_df['genre'])
tf_id.get_feature_names_out()
```
output: 
```python
array(['action', 'adventure', 'ai', 'arts', 'cars', 'comedy', 'dementia',
       'demons', 'drama', 'ecchi', 'fantasy', 'fi', 'game', 'harem',
       'hentai', 'historical', 'horror', 'josei', 'kids', 'life', 'magic',
       'martial', 'mecha', 'military', 'music', 'mystery', 'of', 'parody',
       'police', 'power', 'psychological', 'romance', 'samurai', 'school',
       'sci', 'seinen', 'shoujo', 'shounen', 'slice', 'space', 'sports',
       'super', 'supernatural', 'thriller', 'vampire', 'yaoi', 'yuri'],
      dtype=object)
 ```
* fit_tranform dan pengecekan ukuran
```python
tfidf_matrix = tf_id.fit_transform(anime_df['genre'])
tfidf_matrix.shape # Melihat ukuran matrix tfidf
 ```
Hasilnya (12017, 47)

Berdasarkan output diatas, dapat dilihat bahwa ukuran matriksnya sebesar 12017 x 47

* to_dense()
  
```python
tfidf_matrix.todense()
```
```python
matrix([[0.        , 0.        , 0.        , ..., 0.        , 0.        ,
         0.        ],
        [0.29498527, 0.3162867 , 0.        , ..., 0.        , 0.        ,
         0.        ],
        [0.2516182 , 0.        , 0.        , ..., 0.        , 0.        ,
         0.        ],
        ...,
        [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
         0.        ],
        [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
         0.        ],
        [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
         0.        ]])
```
Berdasarkan output diatas, proses operasi menggunakan todense() sudah berhasil dilakukan

* Pembuatan dataframe dari matrix tf-idf

```python
# Membuat dataframe untuk melihat tf-idf matrix
pd.DataFrame(
    tfidf_matrix.todense(),
    columns=tf_id.get_feature_names_out(),
    index=anime_df.name
).sample(17, axis=1).sample(7, axis=0)
```
Dataframe berhasil dibuat dengan data dari matriks yang sudah dibuat sebelumnya


# Modeling and Result


## Content-Based Filtering
### Modeling

Content-Based Filtering adalah metode yang digunakan dalam sistem rekomendasi untuk memberikan saran kepada pengguna berdasarkan item-item yang telah mereka sukai atau pilih sebelumnya. Metode ini berfokus pada karakteristik atau konten dari item yang ingin direkomendasikan.

Kelebihan Content-Based Filtering:

* Personalisasi: Dapat memberikan rekomendasi yang sangat personal karena didasarkan pada preferensi sebelumnya dari pengguna itu sendiri.
* Transparansi: Mudah untuk menjelaskan mengapa suatu item direkomendasikan, karena rekomendasi didasarkan pada fitur-fitur item yang telah disukai pengguna.

Kekurangan Content-Based Filtering:

* Keterbatasan Diversifikasi: Cenderung merekomendasikan item yang mirip dengan yang sudah diketahui pengguna, sehingga kurang memberikan kejutan atau item baru yang berbeda.
* Ketergantungan pada Konten: Memerlukan data yang cukup tentang konten item untuk bekerja dengan baik, dan kualitas rekomendasi sangat bergantung pada kualitas deskripsi item tersebut.

Pendekatan ini menggunakan atribut-atribut atau fitur-fitur item untuk menentukan kesamaan antara item yang ada. Dalam konteks proyek ini, content-based filtering akan memberikan rekomendasi buku berdasarkan genre dari anime yang ada. Model akan memberikan rekomendasi buku yang memiliki author yang sama.
dimulai dari Proses perhitungan cosine_similarity.

**Berikut ini adalah proses Modelling and Result dari kedua algoritma tersebut:**

* cosine_similarity()
```python
# Proses perhitungan cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim
```

```python
array([[1.        , 0.14715318, 0.        , ..., 0.        , 0.        ,
        0.        ],
       [0.14715318, 1.        , 0.17877808, ..., 0.        , 0.        ,
        0.        ],
       [0.        , 0.17877808, 1.        , ..., 0.        , 0.        ,
        0.        ],
       ...,
       [0.        , 0.        , 0.        , ..., 1.        , 1.        ,
        1.        ],
       [0.        , 0.        , 0.        , ..., 1.        , 1.        ,
        1.        ],
       [0.        , 0.        , 0.        , ..., 1.        , 1.        ,
        1.        ]])
```
 
 Berdasarkan output diatas, proses perhitungan cosine_similarity telah berhasil dilakukan.

* Pembuatan dataframe dari cosine_sim

```python
 # Membuat dataframe dari variabel cosine_sim
cosine_sim_df = pd.DataFrame(cosine_sim, index=anime_df['name'], columns=anime_df['name'])
print('Ukuran Dataframe : ', cosine_sim_df.shape)
```
Ukuran Dataframe :  (12017, 12017)

Berdasarkan output diatas, proses pembuatan dataframe berhasil dilakukan dan dataframe memiliki ukuran 12017 x 12017.

* Similarity matrix pada data
```python
# Melihat similarity matrix pada data
cosine_sim_df.sample(5, axis=1).sample(7, axis=0)
```
* Pembuatan function anime_recommendations()

```python
def anime_recommendations(name, similarity_data=cosine_sim_df, items=anime_df[['name', 'genre']], k=5):
    index = similarity_data.loc[:,name].to_numpy().argpartition(range(-1, -k, -1))
    closest_data = similarity_data.columns[index[-1:-(k+2):-1]]
    closest_data = closest_data.drop(name, errors='ignore')

    return pd.DataFrame(closest_data).merge(items).head(k)

```
Function utama yang digunakan untuk pembuatan model Content Based telah berhasil dibuat.

### Result

```python
anime_df[anime_df.name.eq('Naruto')]
```

||anime_id| 	name| 	genre 	|type| 	episodes| 	rating| 	members|
|---|-------|--------|-------|--------|---------|--------|-------|
|841 	|20| 	Naruto 	|Action, Comedy, Martial Arts, Shounen, Super P... |	TV| 	220| 	7.81| 	683297|

Untuk contoh atau simulasi penggunaan model, kita gunakan naruto yang ber-genre Action, Comedy, Martial Arts, Shounen, Super P...

```python
recommendations_result = anime_recommendations('Naruto')
recommendations_result
```
||name |	genre|
|----|------|-----|
|0 	|Naruto: Shippuuden Movie 4 - The Lost Tower 	|Action, Comedy, Martial Arts, Shounen, Super P...|
|1 |	Naruto Shippuuden: Sunny Side Battle 	|Action, Comedy, Martial Arts, Shounen, Super P...|
|2 	|Boruto: Naruto the Movie - Naruto ga Hokage ni... 	|Action, Comedy, Martial Arts, Shounen, Super P...|
|3 |	Naruto x UT 	|Action, Comedy, Martial Arts, Shounen, Super P...|
|4 	|Naruto: Shippuuden |	Action, Comedy, Martial Arts, Shounen, Super P...|

Berikut ini adalah hasil dari Top-N Recommendation menggunakan Content-Based Filterting. Proses penggunaan model berhasil dilakukan dan model dapat memberikan hasil rekomendasi berdasarkan input yang diberikan.
Pada contoh diatas, model berhasil memberikan rekomendasi anime yang juga ber-genre Action, comedy, martial berdasarkan input yang diberikan, yaitu Naruto yang juga bergenre Action, comedy, martial
**Model telah dapat berfungsi dengan baik.**

## Collaborative Filtering

### Modeling

* Pembuatan class RecommenderNet

```python
class RecommenderNet(Model):

  def __init__(self, num_users, num_anime, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_anime = num_anime
    self.embedding_size = embedding_size

    self.user_embedding = layers.Embedding(
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(num_users, 1)

    self.anime_embedding = layers.Embedding(
        num_anime,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )

    self.anime_bias = layers.Embedding(num_anime, 1)

  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0])
    user_bias = self.user_bias(inputs[:, 0])
    anime_vector = self.anime_embedding(inputs[:, 1])
    anime_bias = self.anime_bias(inputs[:, 1])

    dot_user_anime = tensorflow.tensordot(user_vector, anime_vector, 2)

    x = dot_user_anime + user_bias + anime_bias

    return tensorflow.nn.sigmoid(x)
```
Function utama yang digunakan untuk pembuatan model Collaborative Filtering telah berhasil dibuat

* Inisiasi Model
  
```python
model = RecommenderNet(num_users, num_anime, 50) # inisialisasi model
model.compile(
    loss = keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=[keras.metrics.RootMeanSquaredError()]
)
```
Inisiasi model telah berhasil dilakukan

* Early Stopper
```python
early_stopper = EarlyStopping(monitor='val_root_mean_squared_error',
                              patience=5,
                              verbose=1,
                              restore_best_weights=True)
```
Inisiasi Callback Early Stopper yang akan memantau proses training model. Model akan berhenti jika val_root_mean_squared_error tidak mengalami penurunan lagi selama 5 epochs. Setelah berhenti, model pada epoch tertentu yang memiliki performa terbaik akan dipertahankan.

* Training
```python
history = model.fit(
          x = x_train,
          y = y_train,
          batch_size = 8,
          epochs = 100,
          callbacks = [early_stopper],
          validation_data = (x_val, y_val)
)
```
Berikut ini hasil proses training yang sudah selesai pada epochs ke-11 yang memiliki :
   * loss : 0.5163
   * root_mean_squared_error : 0.1302
   * val_loss : 0.5211
   * val_root_mean_squared_error : 0.1358


```python
Epoch 11/100
8413/8413 ━━━━━━━━━━━━━━━━━━━━ 82s 5ms/step - loss: 0.5163 - root_mean_squared_error: 0.1302 - val_loss: 0.5211 - val_root_mean_squared_error: 0.1385
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 6.
```
### Result

```python
user_id = rating_df.user_id.sample(1).iloc[0]
anime_reviewed_by_user = rating_df[rating_df.user_id == user_id]
anime_not_reviewed = anime_df[~anime_df['anime_id'].isin(anime_reviewed_by_user.anime_id.values)]['anime_id']
anime_not_reviewed = list(
    set(anime_not_reviewed)
    .intersection(set(anime_to_anime.keys()))
)
anime_not_reviewed = [[anime_to_anime.get(x)] for x in anime_not_reviewed]
user_encoder = user_to_user.get(user_id)
user_anime_array = np.hstack(
    ([[user_encoder]] * len(anime_not_reviewed), anime_not_reviewed)
)
     

rating = model.predict(user_anime_array).flatten()

top_rating_indices = rating.argsort()[-10:][::-1]
recommended_anime_ids = [
    anime_encode_to_anime.get(anime_not_reviewed[x][0]) for x in top_rating_indices
]

print('List recommendations anime untuk users : {}'.format(user_id))
print('====' * 9)
print('Anime dengan skor review tinggi dari user ')
print('=====' * 8)

top_anime_user = (
    anime_reviewed_by_user.sort_values(
        by = 'rating',
        ascending=False
    )
    .head(5)
    .anime_id.values
)

anime_df_rows = anime_df[anime_df['anime_id'].isin(top_anime_user)]
for row in anime_df_rows.itertuples():
    print(row.name, ':', row.genre)

print('====' * 8)
print('Top 10 anime recommendation')
print('====' * 8)

recommended_anime = anime_df[anime_df['anime_id'].isin(recommended_anime_ids)]
for row in recommended_anime.itertuples():
    print(row.name, ':', row.genre)
```

```python
28/128 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step
List recommendations anime untuk users : 492
====================================
Anime dengan skor review tinggi dari user 
========================================
One Punch Man : Action, Comedy, Parody, Sci-Fi, Seinen, Super Power, Supernatural
Ookami to Koushinryou II : Adventure, Fantasy, Historical, Romance
Dragon Ball Z : Action, Adventure, Comedy, Fantasy, Martial Arts, Shounen, Super Power
Darker than Black: Kuro no Keiyakusha : Action, Mystery, Sci-Fi, Super Power
Sword Art Online : Action, Adventure, Fantasy, Game, Romance
================================
Top 10 anime recommendation
================================
Haibane Renmei : Drama, Fantasy, Mystery, Psychological, Slice of Life
Gantz 2nd Stage : Action, Drama, Horror, Psychological, Sci-Fi, Supernatural
Gantz : Action, Drama, Horror, Psychological, Sci-Fi, Supernatural
Catand#039;s Eye : Action, Adventure, Comedy, Mystery, Romance
Nissan Serena x One Piece 3D: Mugiwara Chase - Sennyuu!! Sauzando Sanii-gou : Comedy, Fantasy, Shounen
Kuro no Sumika: Chronus : Psychological
Mayoi Neko Overrun! Specials : Comedy, Ecchi
Gilgamesh : Drama, Fantasy, Sci-Fi, Supernatural
Rhea Gall Force : Action, Mecha, Military, Sci-Fi
Shoujo Sect : Comedy, Hentai, Romance, Yuri     
```

Berikut ini adalah hasil dari Top-N Recommendation menggunakan Collaborative Filterting. Proses penggunaan model berhasil dilakukan dan model dapat memberikan hasil rekomendasi berdasarkan rating dari user tertentu dan memberikan rekomendasi anime lainnya yang cocok untuk user tersebut.

Pada contoh diatas, model berhasil memberikan rekomendasi film untuk user nomor 18731 yang pernah memberikan skor rating tinggi ke film dan genre:

   * One Punch Man : Action, Comedy, Parody, Sci-Fi, Seinen, Super Power, Supernatural
   * Ookami to Koushinryou II : Adventure, Fantasy, Historical, Romance
   * Dragon Ball Z : Action, Adventure, Comedy, Fantasy, Martial Arts, Shounen, Super Power
   * Darker than Black: Kuro no Keiyakusha : Action, Mystery, Sci-Fi, Super Power
   * Sword Art Online : Action, Adventure, Fantasy, Game, Romance

Model memberikan 10 rekomendasi berupa film dengan genre:

   * Haibane Renmei : Drama, Fantasy, Mystery, Psychological, Slice of Life
   * Gantz 2nd Stage : Action, Drama, Horror, Psychological, Sci-Fi, Supernatural
   * Gantz : Action, Drama, Horror, Psychological, Sci-Fi, Supernatural
   * Catand#039;s Eye : Action, Adventure, Comedy, Mystery, Romance
   * Nissan Serena x One Piece 3D: Mugiwara Chase - Sennyuu!! Sauzando Sanii-gou : Comedy, Fantasy, Shounen
   * Kuro no Sumika: Chronus : Psychological
   * Mayoi Neko Overrun! Specials : Comedy, Ecchi
   * Gilgamesh : Drama, Fantasy, Sci-Fi, Supernatural
   * Rhea Gall Force : Action, Mecha, Military, Sci-Fi
   * Shoujo Sect : Comedy, Hentai, Romance, Yuri

Model telah dapat berfungsi dengan cukup baik.

# Evaluation


Untuk mengukur bagaimana performa dari model yang telah dibuat, diperlukannya metriks evaluasi untuk mengevaluasi model sistem rekomendasi anime. Berikut adalah rincian metrik yang digunakan untuk tiap pendekatan:

- `Content-Based Filtering` : `Precision`
- `Collaborative Filtering` : `Root Mean Squared Error`

Berikut ini adalah penjelasan mengenai setiap metrik beserta hasil perhitungan metrik dari model yang telah dibuat :

## Content-Based Filtering

 - `Precision
 
     Presisi merupakan ukuran yang menilai efektivitas model klasifikasi dalam mengidentifikasi label positif. Ukuran ini merupakan perbandingan antara jumlah prediksi yang benar-benar positif dengan keseluruhan hasil yang diprediksi sebagai positif, termasuk yang sebenarnya negatif.

    Berikut adalah formula dan cara kerja dari `Precision` :
    
    * **Formula**

       $$Precision = TP/(TP+FP)$$

       Dalam Konteks sistem rekomendasi menjadi:

       ![image](https://github.com/user-attachments/assets/b04447ad-a744-4d7c-b5c8-70083c3124ef)

       Gambar 5 - Formula Precision
      
   * **Cara Kerja**

     Formula tersebut mengukur presisi dalam konteks sistem rekomendasi. Presisi dihitung dengan membagi jumlah rekomendasi yang relevan dengan jumlah total item yang    direkomendasikan. Jadi, jika sebuah sistem merekomendasikan 10 film dan hanya 6 yang relevan atau disukai oleh pengguna, maka presisi sistem tersebut adalah 0.6 atau 60%. Ini menunjukkan seberapa akurat sistem dalam memberikan rekomendasi yang sesuai dengan kebutuhan atau selera pengguna.
      
  - Penjelasan Hasil `Precision` dari model `Content-Based Learning`
     - Fungsi dari `calculate_precision` digunakan untuk perhitungan Presisi berdasarkan formula Presisi

 
```python
# Calculate precision based on title and genre
def calculate_precision(name, genre):
    name_genre_anime = anime_df[(anime_df['name'] ==name) & (anime_df['genre'] == genre)]
    recommended_animes = animes_recommendations(name)
    relevant_animes = recommended_animes[(recommended_animes['genre'] == genre)]
    precision = len(relevant_animes['genre'] == genre) / len(recommended_animes['genre'] == genre)

    return precision
```
```python
print(f'The precision of the recommendation system is {precision:.1%}')
```
The precision of the recommendation system is 100.0%

Dari hasil rekomendasi bagian result di atas, diketahui bahwa Naruto termasuk ke dalam genre (Action, Comedy, Martial Arts, Shounen, Super P) Dari 5 item yang direkomendasikan, 5 item memiliki genre (Action, Comedy, Martial Arts, Shounen, Super P). Precision = TP/(TP+FP) Dalam Konteks sistem rekomendasi menjadi:

Ini sesuai dengan formula tunjukan P recision = #of recommendation that are relevant/#of item we recommend. Pada contoh rekomendasi resto di atas: Precission = 5/5. **Jadi presisinya = 100%** 

**Model memiliki performa yang sangat baik dalam memberikan rekomendasi secara Content-Based Filtering.**


## Collaborative Filtering

  - `Root Mean Squared Error`
    
    Root Mean Square Error (RMSE) adalah metrik yang sering digunakan dalam machine learning untuk mengukur seberapa baik sebuah model prediktif dapat memperkirakan nilai yang sebenarnya. RMSE merupakan akar kuadrat dari rata-rata perbedaan kuadrat antara nilai yang diprediksi oleh model dan nilai yang sebenarnya (nilai aktual).

    Berikut ini adalah formula dan cara kerja dari `Root Mean Squared Error` :

    - **Formula**
   
      $$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
   
    - **Cara Kerja**
      
      RMSE menghitung akar kuadrat dari rata-rata perbedaan kuadrat antara nilai yang diprediksi oleh model dan nilai sebenarnya. Proses kerjanya melibatkan beberapa langkah. Pertama, untuk setiap titik data, kita menghitung selisih antara prediksi model dan nilai aktual. Selisih ini kemudian dikuadratkan untuk menghilangkan nilai negatif dan memberikan bobot lebih pada kesalahan yang lebih besar. Setelah itu, kita menghitung rata-rata dari nilai-nilai kuadrat tersebut. Terakhir, kita mengambil akar kuadrat dari rata-rata ini untuk mendapatkan RMSE.
    
  - Penjelasan Hasil `Root Mean Squared Error` dari model `Collaborative Learning`
```python
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model_metrics')
plt.ylabel('root_mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
```
![Untitled](https://github.com/user-attachments/assets/4f006ad3-4a0f-4d77-bef9-09fb2303c2b5)
Gambar 6. plot evaluasi Collaborative Filtering

Berdasarkan plot tersebut, proses training model berhenti pada epoch ke 11 (epochs 1 dimulai dari nomor 0 pada plot) karena callbacks yang berisi early stopper. early stopper menghentikan proses training karena model tidak menunjukkan penurunan yang lebih keci dari val_root_mean_squared_error pada epochs ke-11 selama 6 epochs berturut-turut.

Kemudian, model pada epochs ke 11 yang dipertahankan karena pada epochs tersebut model memiliki performa yang terbaik. Berikut adalah hasil dari metriks pada epocs tersebut:

   * loss : 0.5163
   * root_mean_squared_error : 0.1302
   * val_loss : 0.5211
   * val_root_mean_squared_error : 0.1358

    
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
