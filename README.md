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
