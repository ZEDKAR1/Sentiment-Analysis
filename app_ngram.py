import streamlit as st
import base64
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
from nltk.corpus import stopwords
nltk.download('stopwords')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix
import seaborn as sns
from collections import Counter
from nltk.util import ngrams



st.set_page_config(layout='wide')

# Fungsi untuk menload image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_path = 'asset/background.jpeg' 
img_base64 = get_base64_of_bin_file(img_path)

# Set up background website
page_bg_img = f'''
<style>
.stApp {{
    background-image: linear-gradient(rgba(9,0,77,0.75),rgba(9,0,77,0.75)), url("data:image/jpg;base64,{img_base64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
</style>
'''

hover = """
<style>
[data-testid="stBaseButton-secondary"] {
    padding:9px 25px;
    background-color: rgba(0,136,169,1);
    border: none;
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.3s ease 0s;
}

[data-testid="stBaseButton-secondary"]:hover {
    background-color: rgba(0,136,169,0.8);
}
</style>
"""

# Apply the custom CSS
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown(hover, unsafe_allow_html=True)

if 'predict' not in st.session_state:
    st.session_state.predict = 'false' #prediksi/evaluasi

if 'page' not in st.session_state:
    st.session_state.page = 'home' #home

if 'tipe' not in st.session_state:
    st.session_state.tipe = 'csv' #tipe data csv/text

if 'proses' not in st.session_state:
    st.session_state.proses = 'false' #about model

def home_page():
    st.session_state.back = st.session_state.page
    st.session_state.page = 'home'

def input_page():
    st.session_state.back = st.session_state.page
    st.session_state.page = 'input'

def preprocessing_page():
    st.session_state.back = st.session_state.page
    st.session_state.page = 'preprocessing'

def output_page():
    st.session_state.back = st.session_state.page
    st.session_state.page = 'output'

def about_page():
    st.session_state.back = st.session_state.page
    st.session_state.page = 'about'

def about_model():
    st.session_state.back = st.session_state.page
    st.session_state.page = 'model'

def back():

    st.session_state.page = st.session_state.back

norm = {
    'serusaya': 'seru saya',
    'gimnya' : 'game nya',
    'byk': 'banyak',
    'terimakasih': 'terima kasih',
    'devloper': 'developer',
    'monton': 'moonton',
    'thank': 'terima kasih',
    'thanks': 'terima kasih',
    'dev': 'developer',
    'god': 'bagus',
    'mabar': 'main bareng',
    'ny': 'nya',
    'jga': 'juga',
    'ga': 'tidak',
    'dangk': 'dan tidak',
    'gk': 'tidak',
    'sat': 'saat',
    'yg': 'yang',
    'kang': 'abang',
    'deh': 'dah',
    'bwangwet': 'banget',
    'ni': 'ini',
    'bet': 'banget',
    'gravik': 'grafik',
    'geme': 'game',
    'mantab': 'mantap',
    'engak': 'tidak',
    'segansegan': 'segan',
    'colabepic': 'collab epic',
    'mainya': 'mainnya',
    'enjoy': 'nyaman',
    'xixixixi': 'hahaha',
    'g': 'tidak',
    'skinya': 'skinnya',
    'gem': 'game',
    'gak': 'tidak',
    'sru': 'seru',
    'bru': 'baru',
    'apstore': 'appstore',
    'kikir': 'pelit',
    'legendaris': 'legends',
    'sgat': 'sangat',
    'games': 'game',
    'ok': 'oke',
    'freze' : 'freeze',
    'bgt': 'banget',
    'bgtbanyak': 'banget banyak',
    'kern': 'keren',
    'bgus': 'bagus',
    'gokil': 'keren',
    'nice': 'bagus',
    'fre': 'gratis',
    'tau': 'tahu',
    'smoth': 'halus',
    'bgs': 'bagus',
    'gam': 'game',
    'gil': 'gila',
    'nih': 'ini',
    'hehehehe': 'hahaha',
    'kren': 'keren',
    'kedepanya': 'kedepannya',
    'bagu': 'bagus',
    'downlond': 'unduh',
    'wow': 'luar biasa',
    'mantapmenghibur ': 'mantap menghibur',
    'say': 'saya',
    'sagat': 'sangat',
    'mlb': 'mlbb',
    'menyenang': 'menyenangkan',
    'kanmantap': 'mantap',
    'tuk': 'untuk',
    'gemnya': 'gamenya',
    'anjir': 'anjing',
    'min': 'developer',
    'memainkanya': 'memainkannya',
    'sukak': 'suka',
    'semagat': 'semangat',
    'pokonya': 'pokoknya',
    'kaya': 'seperti',
    'mennag' : 'menang',
    'mnang' : 'menang',
    'kayak': 'seperti',
    'mengumkan': 'mengagumkan',
    'b': 'bintang',
    'rekomendet': 'direkomendasikan',
    'gacor': 'keren',
    'toxsi': 'kasar',
    'ngk': 'tidak',
    'kyk': 'seperti',
    'skin²': 'skin',
    'gameny': 'gamenya',
    'baikm': 'baik',
    'dpat': 'dapat',
    'hero': 'hero',
    'bngt': 'banget',
    'lumyan': 'lumayan',
    'gemenya': 'gamenya',
    'ska': 'suka',
    'lainya': 'lainnya',
    'mobilegend': 'mobile legends',
    'okeh': 'oke',
    'kek': 'seperti',
    'ges': 'teman',
    'bangt': 'banget',
    'trus': 'terus',
    'mantep': 'mantap',
    'drak': 'dark',
    'sytem': 'system',
    'sep': 'oke',
    'nyakarena': 'nya karena',
    'mudahsaya': 'mudah saya',
    'diamondsemoga': 'diamond semoga',
    'eventevent': 'event',
    'aja': 'saja',
    'gamenyaaku': 'gamenya aku',
    'leg': 'lag',
    'nyah': 'nya',
    'lh': 'lah',
    'grafis': 'grafik',
    'mantao': 'mantap',
    'bro': 'kawan',
    'mantp': 'mantap',
    'jlas': 'jelas',
    'budeg': 'tidak mendengar',
    'kaga': 'tidak',
    'menangapi': 'menganggapi',
    'capek': 'lelah',
    'dm': 'diamond',
    'gilira': 'sedangkan',
    'giliran': 'sedangkan',
    'musu': 'musuh',
    'sosoan': 'sok',
    'ful': 'penuh',
    'dragsistem': 'dark system',
    'jaringansebab': 'jariingan sebab',
    'tingi': 'tinggi',
    'oke²aja': 'oke saja',
    'gj': 'tidak jelas',
    'ush': 'usah',
    'sistem': 'system',
    'gema': 'game',
    'gue': 'gw',
    'reng': 'rank',
    'susahlol': 'susah',
    'tin': 'tim',
    'sory': 'maaf',
    'uninstal': 'uninstall',
    'tpi': 'tapi',
    'ngeleg': 'ngelag',
    'ajaudah': 'saja sudah',
    'tolongsudah': 'tolong sudah',
    'stabilpadahal': 'stabil padahal',
    'kencengtapi': 'kencang tapi',
    'ngelagtolong': 'ngelag tolong',
    'draksistem': 'dark system',
    'jaringanya': 'jaringannya',
    'trlalu': 'terlalu',
    'pakek': 'pakai',
    'ngelak': 'ngelag',
    'gamae': 'game',
    'tu': 'itu',
    'dar': 'dark',
    'darksystemnya': 'dark system nya',
    'eror': 'error',
    'efect': 'efek',
    'klo': 'kalau',
    'loby': 'beranda',
    'logind': 'masuk',
    'maen': 'main',
    'lgsg': 'langsung',
    'pula': 'juga',
    'maenya': 'mainnya',
    'niru' : 'plagiat',
    'meniru' : 'plagiat',
    'pelagiat': 'plagiat',
    'guoblok': 'goblok',
    'm': 'mm',
    'ajg': 'anjing',
    'kopi': 'niru',
    'plagiathapus': 'plagiat hapus',
    'tol': 'kontol',
    'c': 'cc',
    'nyengol': 'nyenggol',
    'sebelahtolong': 'sebelah tolong',
    'menjelekjelekan': 'menjelekkan',
    'respek': 'hormat',
    'gajelas': 'tidak jelas',
    'anj': 'anjing',
    'trol': 'troll',
    'muluarcher': 'selalu archer',
    'kalahmenang': 'kalah menang',
    'kalahyg': 'kalah yang',
    'turunmain': 'turun main',
    'lagend': 'legend',
    'tak': 'tidak',
    'd': 'di',
    'oada': 'pada',
    'poinya': 'poinnya',
    'mainmau': 'main mau',
    'ajh': 'saja',
    'bat': 'banget',
    'mengangu': 'mengganggu',
    'dowld': 'download',
    'pdh': 'padahal',
    'brp': 'berapa',
    'apl': 'aplikasi',
    'sistemnyacape': 'systemnya cape',
    'buajingan': 'bajingan',
    'ku': 'aku',
    'stak': 'nyangkut',
    'dlu': 'dulu',
    'tetep': 'tetap',
    'gjls': 'tidak jelas',
    'pdhal': 'padahal',
    'asasin': 'assasin',
    'menyusahkantidak': 'menyusahkan tidak',
    'perbedaannyasama': 'perbedaannya sama',
    'blon': 'bodoh',
    'gtu': 'begitu',
    'enganya': 'engganya',
    'ben': 'beban',
    'heroskin': 'hero skin',
    'buriq': 'burik',
    'gamestolong': 'game tolong',
    'dinerfkarena': 'dinerf karena',
    'apaplkarena': 'apa karena',
    'anjdi': 'anjing di',
    'gd': 'tidak ada',
    'kntl': 'kontol',
    'anjg': 'anjing',
    'ngeleg²': 'ngelag',
    'apan': 'apaan',
    'spek': 'spesifikasi',
    'batle': 'perangg',
    'asw': 'anjing',
    'maf': 'maaf',
    'riloadkeluar': 'reload keluar',
    'mach': 'match',
    'rom': 'room',
    'oklah': 'oke lah',
    'ngak': 'tidak',
    'lagak': 'tidak',
    'masak': 'masa',
    'ajamatchmaking': 'saja matchmaking',
    'para': 'parah',
    'emel': 'ml',
    'as': 'anjing',
    'kntol': 'kontol',
    'ngetrol': 'ngetroll',
    'bayakan': 'banyakan',
    'f': 'ff',
    'mao': 'mau',
    'knp': 'kenapa',
    'sentiasa': 'senantiasa',
    'tolololol': 'tolol',
    'gm': 'game',
    'kadang²': 'kadang',
    'ngelek': 'ngelag',
    'myesel': 'kesal',
    'bocil': 'bocah kecil',
    'bnyak': 'banyak',
    'ngejiplak': 'plagiat',
    'pdhl': 'padahal',
    'sebelahmlb': 'sebelah mlbb',
    'dri': 'dari',
    'pda': 'pada',
    'bayak': 'banyak',
    'loding': 'loading',
    'kenceng': 'kencang',
    'permintan': 'permintaan',
    'satutolong': 'satu tolong',
    'diperbaikinfps': 'diperbaiki fps',
    'rankdan': 'rank dan',
    'namah': 'nama',
    'cinacoba': 'china coba',
    'bebepara': 'beberapa',
    'cina': 'china',
    'angota': 'anggota',
    'ama': 'sama',
    'frez': 'freeze',
    'iklanya': 'iklannya',
    'penjelasanya': 'penjelasannya',
    'tdk': 'tidak',
    'anjim': 'anjing',
    'geblek': 'goblok',
    'losetreak': 'losestreak',
    'kerjan': 'kerjaan',
    'gebleg': 'goblok',
    'jgn': 'jangan',
    'ngefrane': 'ngeframe',
    'folow': 'ikut',
    'fidio': 'vidio',
    'tlol': 'tolol',
    'warasmasa': 'waras masa',
    'diapainditambah': 'diapain ditambah',
    'ngoptimalkan': 'mengoptimalkan',
    'susahpakai': 'susah pakai',
    'hutankocak': 'hutan kocak',
    'koplak': 'kocak',
    'nob': 'payah',
    'bangat' : 'banget',
    'elek': 'jelek',
    'banga': 'bangga',
    'bae': 'aja',
    'mnding': 'mending',
    'minuntuk': 'developer untuk',
    'pecahkansama': 'pecahkan sama',
    'otomatispaling': 'otomatis paling',
    'susuah': 'susah',
    'gamenyabaru': 'gamenya baru',
    'gamenyapadahal': 'gamenya padahal',
    'dmn': 'dimana',
    'pd': 'pada',
    'patahsinyal': 'patah sinyal',
    'tibatiba': 'tiba',
    'aru': 'baru',
    'lgi': 'lagi',
    'yabiar': 'ya biar',
    'mainyasama': 'mainnya sama',
    'kalu': 'kalau',
    'menantangsatu': 'menantang satu',
    'suman': 'cuman',
    'sihgrafik': 'sih grafik',
    'iyo': 'iya',
    'terbitkasian': 'terbit kasian',
    'nga': 'tidak',
    'ngebag': 'ngebug',
    'ajah': 'saja',
    'manding': 'mending',
    'stafstaf': 'staf',
    'kont': 'kontol',
    'ngeleq': 'ngelag',
    'gaje': 'tidak jelas',
    'goblog': 'goblok',
    'aga': 'agak',
    'benarbenar': 'benar',
    'setres': 'setress',
    'darksistem': 'dark system',
    'mlbedakelasbedakualitas': '',
    'nyingung': 'nyinggung',
    'asu': 'anjing',
    'dwld': 'download',
    'kagak': 'tidak',
    'ngestack': 'ngestuck',
    'tampat': 'tempat',
    'cmn': 'cuman',
    'lalukek': 'lalu seperti',
    'sm': 'sama',
    'diamon': 'diamond',
    'ngestak': 'ngestuck',
    'matakata': 'mata kata',
    'yasiapa': 'ya siapa',
    'lagih': 'lagi',
    'judescuek': 'judes cuek',
    'pengunan': 'penggunaan',
    'bnget': 'banget',
    'semuagame': 'semua game',
    'kakulahnyaga': 'kaku lagnya tidak',
    'mainmasa': 'main masa',
    'gamegame': 'game',
    'sbelah': 'sebelah',
    'awak': 'awal',
    'gerakanya': 'gerakannya',
    'blok': 'goblok',
    'mingu': 'minggu',
    'kekgini': 'kayak begini',
    'emng': 'emang',
    'jaringansoalnya': 'jaringan soalnya',
    'mainpingnya': 'main pingnya',
    'capekintinya': 'lelah intinya',
    'baikitu': 'baik itu',
    'ngeleq' : 'ngelag',
    'stuntombol': 'stuntombol',
    'setalah': 'setelah',
    'stuk': 'stuck',
    'jengkelmemuat': 'kesal loading',
    'jengkel' : 'kesal',
    'memuat' : 'loading',
    'stack': 'stuck',
    'legen': 'legends',
    'ak': 'aku',
    'sekalikalah': 'sekali kalah',
    'sesesui': 'sesuai',
    'karna': 'karena',
    'gax': 'tidak',
    'memengkan': 'memenangkan',
    'brkurang': 'berkurang',
    'tangapi': 'tanggapi',
    'kurangloby': 'kurang beranda',
    'legens': 'legends',
    'bangr': 'banget',
    'mkn': 'makan',
    'ngfrane': 'ngeframe',
    'bnr': 'benar',
    'ketingalan': 'ketinggalan',
    'matchmaching': 'match making',
    'majapahitkaku': 'majapahit kaku',
    'bong': 'bohong',
    'guna': 'berguna',
    'sma': 'sama',
    'udh': 'udah',
    'game nya': 'gamenya',
    'darksystem': 'dark system',
    'riload': 'reload',
    'kesukan': 'kesukaan',
    'gemanya': 'gamenya',
    'mobilr': 'mobile',
    'hadeh': 'aduh',
    'ngabissin': 'ngehabisin',
    'lo': 'lu',
    'ntap': 'mantap',
    'legand': 'legend',
    'seberu': 'seru',
    'mantaf': 'mantap',
    'dabest': 'terbaik',
    'bersemqngat': 'bersemangat',
    'feder': 'feeder',
    'ilang': 'hilang',
    'asek': 'asik',
    'mantul': 'mantap betul',
    'pnyakitnya': 'penyakitnya',
    'sebelumnyamlserver': 'sebelumnya ml server',
    'cacad': 'cacat',
    'develover': 'developer',
    'anjinx': 'anjing',
    'muludapat': 'mulu dapat',
    'ampasnaik': 'ampas naik',
    'tertingi': 'tertinggi',
    'geratis': 'gratis',
    'ifen': 'event',
    'pintng': 'bintang',
    'jdi': 'jadi',
    'tencentbaek': 'tencent baik',
    'fed': 'feed',
    'cmantep': 'mantap',
    'nij': 'ini',
    'bagusgrafik': 'bagus grafik',
    'lancarhero': 'lancar hero',
    'demage': 'damage',
    'turet': 'turret',
    'tp': 'tapi',
    'lemot': 'lama',
    'kasi': 'kasih',
    'kntil': 'kontol',
    'bodo': 'bodoh',
    'moga': 'semoga',
    'menaikan': 'menaikkan',
    'ke bayakan': 'kebanyakan',
    'idamanmanjain': 'idaman manjain',
    'kingsmemang': 'kings memang',
    'ngedadak': 'mendadak',
    'systim': 'system',
    'mls': 'malas',
    'best': 'terbaik',
    'serulah': 'seru lah',
    'bagos': 'bagus',
    'deley': 'delay',
    'ofline': 'offline',
    'kok': 'malah',
    'percumah': 'percuma',
    'ap': 'apa',
    'cba': 'coba',
    'kenakan': 'keenakan',
    'dog': 'anjing',
    'mobilegends': 'mobile legends',
    'maenin': 'mainin',
    'rengkednya': 'rankednya',
    'bner': 'benar',
    'marchmackingnya': 'matchmakingnya',
    'bruntun': 'beruntun',
    'mntap': 'mantap',
    'bagus san': 'bagusan',
    'gada': 'tidak ada',
    'reting': 'rating',
    'nyindar': 'nyindir',
    'ngeprime': 'ngeframe',
    'okemackmaking': 'oke matchmaking',
    'nabrakbaru': 'nabrak baru',
    'sebelahmobile': 'sebelah mobile',
    'muda': 'mudah',
    'copy': 'plagiat',
    'tungu': 'tunggu',
    'bagussaya': 'bagus saya',
    'perainanya': 'permainannya',
    'inisalam': 'ini salam',
    'the best': 'terbaik',
    'gp': 'gapapa',
    'kil': 'kill',
    'recal': 'recall',
    'skil': 'skill',
    'kumainin': 'aku mainin',
    'perkembangkanlebih': 'berkembang lebih',
    'perkembangkan': 'berkembang',
    'baikramah': 'baik ramah',
    'dn': 'dan',
    'bnyk': 'banyak',
    'mlh': 'malah',
    'gx': 'tidak',
    'serujuga': 'serujuga',
    'dpt': 'dapat',
    'elu': 'lu',
    'setu': 'seru',
    'jlk': 'jelek',
    'gim' : 'game',
    'gangu': 'ganggu',
    'aj': 'aja',
    'gimna': 'gimana',
    'plagiyat': 'plagiat',
    'slain': 'selain',
    'sagt': 'sangat',
    'lg' : 'lagi',
    'kalo': 'kalau',
    'gemes': 'game',
    'bantuanya': 'bantuannya',
    'muluk': 'mulu',
    'sprt': 'seperti',
    'scren': 'screen',
    'mobilelegend': 'mobile legend',
    'garfik': 'grafik',
    'jagi': 'jago',
    'anjeng' : 'anjing',
    'jngan': 'jangan',
    'moton': 'moonton',
    'lbh': 'lebih',
    'drpd': 'daripada',
    'semingu': 'seminggu',
    'overal': 'keseluruhan',
    'gemanya' : 'gamenya',
    'skli' : 'sekali',
    'ngefreze' : 'freeze',
    'systm': 'system',
    'gemplagiat': 'game plagiat',
    'plgiat' : 'plagiat',
    'ktmu' : 'ketemu',
    'ketmu' : 'ketemu',
    'lumayam': 'lumayan',
    'lumayamlah': 'lumayan lah',
    'ngbug' : 'ngebug',
    'hok' : 'honor of kings',
    'langen': 'legends',
    'palagiat': 'plagiat',
    'matcmaking' : 'matchmaking',
    'ng': 'tidak',
    'jiplak' : 'plagiat',
    'penjiplak' : 'plagiat',
    'sistemnyabikin' : 'system nya membuat',
    'bikin' : 'membuat',
    'daksistem' : 'dark system',
    'dak' : 'dark',
    'sayasistem' : 'saya system',
    'sistemnya' : 'system nya'
 }


lstm = 'model/LSTM_94.keras'
word2vec = 'model/word2vec_v29.w2v'
lstm_2 = 'model/LSTM_2_gram.keras'
lstm_3 = 'model/LSTM_3_gram.keras'
word2vec_2 = 'model/bigram_model.w2v'
word2vec_3 = 'model/trigram_model.w2v'

def normalize_text(text):
    words = text.split()  # Memisahkan kata
    normalized_words = [norm.get(word, word) for word in words]
    return ' '.join(normalized_words)  # Gabungkan kembali menjadi string

# Daftar stopword dari NLTK (bahasa Indonesia)
stop_words = set(stopwords.words('indonesian'))

# Kata yang ingin dikecualikan dari stopword (misalnya 'tidak' dan 'baik' tidak ingin dihapus)
exclude_words = {
    'tidak',
    'baik',
    'bisa',
    'kecil',
    'semuanya',
    'banyak',
    'jelas',
    'mirip',
    'lebih',
    'sangat',
    'semua',
    'jauh',
    'jangan',
    'naik',
    'tepat',
    'luar',
    'baru',
    'dapet',
    'pernah',
    'benar',
    'masalah',
    'buat',
    'sering',
    'punya',
    'sekali',
    'keluar',
    'mulai',
    'percuma'
}

# Kata yang ingin dimasukkan ke dalam daftar stopword
include_words = {
    'nya',
    'gw',
    'gua',
    'lu'
}

# Hapus kata-kata yang dikecualikan dari daftar stopword
stop_words = stop_words.difference(exclude_words)

# Tambahkan kata-kata yang ingin disertakan ke dalam stopword
stop_words = stop_words.union(include_words)

# Fungsi untuk menghapus stopword
# def remove_stopwords(tokens, stop_words):
#     return [token for token in tokens if token not in stop_words]

# Membuat objek stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi stemming
def stemming_indonesia(tokens):
    return [stemmer.stem(token) for token in tokens]

def normalize_word(word):
    return re.sub(r'(.)\1+', r'\1', word)

def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', '', text) # Menghapus emoticon menggunakan regex untuk karakter non-ASCII
    text = re.sub(r'#\w+', '', text) # Menghapus hashtag
    text = re.sub(r'\d+', '', text) # Menghapus angka
    text = re.sub(r'[^\w\s]', '', text) # Menghapus tanda baca
    return text

def generate_bigrams(tokens):
    return list(ngrams(tokens, 2))  # Membentuk bigram dari token yang ada

# Home Page
if st.session_state.page == 'home':
    st.markdown(
    "<div style='font-size:70px;font-weight:bold;'>Selamat Datang,</div>", 
    unsafe_allow_html=True
    )
    st.markdown(
    "<div style='font-size:25px;'>Kami menggunakan kekuatan teknologi LSTM (Long Short-Term Memory) untuk menganalisis sentimen dari teks secara akurat dan mendalam, Alat ini mampu memahami pola-pola dalam data teks dan memberikan hasil analisis sentimen dengan tingkat presisi yang tinggi.</div>", 
    unsafe_allow_html=True
    )
    st.markdown(
    "<div style='font-size:70px;font-weight:bold;'>Apa itu Analisis Sentimen?</div>", 
    unsafe_allow_html=True
    )
    st.markdown(
    "<div style='font-size:25px;margin-bottom:30px;'>Analisis sentimen adalah proses memahami dan mengelompokkan emosi atau opini yang terdapat dalam teks, apakah positif, negatif, atau netral. Dengan metode LSTM, model dilatih untuk menangkap nuansa emosi.</div>", 
    unsafe_allow_html=True
    )

    # Tombol untuk about model
    if st.button("About Model"):
        st.session_state.proses = 'true'
        input_page()
        st.rerun()

    # Tombol untuk evaluasi model
    if st.button("Evaluasi Model"):
        st.session_state.predict = 'false'
        st.session_state.proses = 'false'
        input_page()
        st.rerun()
    
    # Tombol untuk prediksi model
    if st.button("Coba Prediksi"):
        st.session_state.predict = 'true'
        st.session_state.proses = 'false'
        input_page()
        st.rerun()
    
    # Tombol untuk about me
    if st.button("About"):
        st.session_state.proses = 'false'
        about_page()
        st.rerun()
        
# Input Page
elif st.session_state.page == 'input':
    if st.session_state.proses == 'false': # Bukan About Model - Input
        st.title("Halaman Input Data")

        if st.session_state.predict == 'true': # Prediksi
            st.write("Masukkan dataset untuk di prediksi. Pastikan terdapat kolom 'content'.")
        else: # Evaluasi
            st.write("Masukkan dataset untuk mengevaluasi model. Pastikan terdapat kolom 'content' dan 'sentimen'.")
            
        tab1, tab2 = st.tabs(["CSV", "Text"])

        with tab1: # CSV
            uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
            
                st.table(df.head(10))
                st.session_state.df = df

                st.write("File uploaded successfully! Here is a preview of your data:")
                if st.session_state.predict == 'false': # Evaluasi
                    if "sentimen" not in df.columns or 'content' not in df.columns: # Error
                        st.error("❌ Kolom 'sentimen' atau 'content' tidak ditemukan dalam file.")
                    else:
                        if st.button("Process Text") and uploaded_file is not None:
                            st.session_state.tipe = 'csv'
                            preprocessing_page()
                            st.rerun()

                else: # Prediksi
                    if 'sentimen' in df.columns or 'content' not in df.columns: # Error
                        st.error("❌ Hapus kolom 'sentimen' atau tambahkan 'content' dalam file.")
                    else:
                        if st.button("Process Text") and uploaded_file is not None:
                            st.session_state.tipe = 'csv'
                            preprocessing_page()
                            st.rerun()

            else: # Error Handling
                if st.session_state.predict == 'false': # Evaluasi
                    st.write("Silahkan upload file yang memiliki kolom 'content' dan 'sentimen' seperti contoh dibawah.")
                    contoh = pd.read_csv("csv/contoh.csv")
                    st.dataframe(contoh[['content', 'sentimen']], hide_index=True)
                else: # Prediksi
                    st.write("Silahkan upload file yang memiliki kolom 'content' seperti contoh dibawah.")
                    contoh = pd.read_csv("csv/contoh_predik.csv")
                    st.dataframe(contoh['content'], hide_index=True)

        with tab2: # Satuan (text)
            data = []
            st.title("Input your sentiment")

            # Text Input
            text = st.text_input("Enter your text:")

            # Sentiment Input
            if st.session_state.predict == 'false': # Evaluasi
                sentiment = st.selectbox("Select the sentiment:", options=[1, 0], format_func=lambda x: "Positive" if x == 1 else "Negative")
            
            if text is not '':
                if st.session_state.predict == 'false': # Evaluasi
                    data.append({"content": text, "sentimen": sentiment})
                else: # Prediksi
                    data.append({'content':text})
                st.success("Data added!")
                if data:
                    st.subheader("DataFrame:")
                    df = pd.DataFrame(data)
                    st.dataframe(df, hide_index=True)
                    st.session_state.df = df

                    if st.button("Preprocessing"):
                        st.session_state.tipe = 'text' 
                        preprocessing_page()
                        st.rerun()

            else:
                st.error("Please enter some text.")

        
        if st.button("back"):
            back()
            st.rerun()

    else: # About Model - Input
        st.title("Data mentah model yang sudah di label")
        left, right = st.columns([3,1])

        with left:
            df = pd.read_csv("csv/HOK_Labeled.csv")
            df = df[df['sentimen'].isin([1,0])]
            df['sentimen'] = df['sentimen'].replace({1: 'Positif', 0: 'Negatif'})

            st.dataframe(df, hide_index=True)

        with right:
            st.dataframe(df['sentimen'].value_counts())

            if st.button("Next"):
                preprocessing_page()
                st.rerun()

            if st.button("back"):
                back()
                st.rerun()

# Preprocessing Page
elif st.session_state.page == 'preprocessing': 
    if st.session_state.proses == 'false': # Bukan About Model - Preprocessing
        st.title('Halaman Preprocessing')
        st.write('Berikut adalah hasil dari preprocessing dari dataset yang sudah diinputkan:')
        df = st.session_state.df
        df['lower_case'] = df['content'].str.lower()
        df['clean_1'] = df['lower_case'].apply(clean_text)
        df['clean_2'] = df['clean_1'].apply(normalize_word)
        df['normalized'] = df['clean_2'].apply(normalize_text)
        df['tokens'] = df['normalized'].apply(word_tokenize)
        # df['stopword'] = df['tokens'].apply(lambda x:remove_stopwords(x, stop_words))
        df['stemming'] = df['tokens'].apply(stemming_indonesia)

        if st.session_state.tipe == 'csv': # CSV
            if st.session_state.predict == 'false': # Evaluasi
                st.dataframe(df[['content','lower_case','clean_1', 'clean_2', 'normalized', 'tokens', 'stemming','sentimen']], hide_index=True)
            else: # Prediksi
                st.dataframe(df[['content','lower_case','clean_1', 'clean_2', 'normalized', 'tokens', 'stemming']], hide_index=True)

            if st.button("Evaluasi model"):
                st.session_state.df = df
                output_page()
                st.rerun()

        else: # Satuan (text)
            df_transposed = df.transpose()
            st.dataframe(df_transposed, use_container_width=True)
            if st.button("Prediksi!"):
                st.session_state.df = df
                output_page()
                st.rerun()
        
        if st.button("back"):
            back()
            st.rerun()
            
    else: # About Model - Preprocessing
        st.title("Data Setelah Diprerprocessing ")
        df = pd.read_csv('csv/data_pre_new.csv')
        df['sentimen'] = df['sentimen'].replace({2:0})
        st.dataframe(df, hide_index=True)

        if st.button("Next"):
            output_page()
            st.rerun()
        
        if st.button("Back"):
            back()
            st.rerun()

# Output Page
elif st.session_state.page == 'output':
    if st.session_state.proses == 'false': # Bukan About Model - Output
        df = st.session_state.df
        model = load_model(lstm_2)
        word2vec_model = Word2Vec.load(word2vec_2)
        w2v = word2vec_model.wv
        max_sequence_length = 56

        st.title("Output Page")
        st.write("Pada halaman ini akan menampilkan hasil evaluasi model!")
        st.divider()
        left, right = st.columns([2,3])

        df['2_gram'] = [[" ".join(bigram) for bigram in generate_bigrams(tokens)] for tokens in df['stemming']]
        predik = df['2_gram']
        
        def convert_and_pad(data, word2vec_model, maxlen, vector_size):
            vectors = [
                [word2vec_model[word] if word in word2vec_model else np.zeros(vector_size) for word in sentence]
                for sentence in data
            ]
            vectors_padded = pad_sequences(vectors, maxlen=maxlen, padding='post', dtype='float32')
            return vectors_padded

        predik_pad = convert_and_pad(predik, w2v, max_sequence_length, vector_size=128)
        predictions = model.predict(predik_pad)
        df['prediksi'] = ['positif' if pred >= 0.5 else 'negatif' for pred in predictions.flatten()]

        if st.session_state.tipe == 'csv': # CSV
            tombol = 'false'
            if st.session_state.predict == 'false': # Evaluasi
                df['sentimen'] = df['sentimen'].replace({1: 'positif', 0: 'negatif'})
                sentimen_count = df['sentimen'].value_counts()

            predict_count = df['prediksi'].value_counts()

            with left:
                if st.session_state.predict == 'false': # Evaluasi
                    st.markdown(
                    "<div style='font-size:35px;margin-bottom:15px;font-weight:bold;'>Sentimen Asli:</div>", 
                    unsafe_allow_html=True
                    )
                    st.write(f"Jumlah Positif: {sentimen_count.get('positif', 0)}")
                    st.write(f"Jumlah Negatif: {sentimen_count.get('negatif', 0)}")

                st.markdown(
                "<div style='font-size:35px;margin-bottom:15px;font-weight:bold;'>Sentimen Prediksi:</div>", 
                unsafe_allow_html=True
                )
                st.write(f"Jumlah Positif: {predict_count.get('positif', 0)}")
                st.write(f"Jumlah Negatif: {predict_count.get('negatif', 0)}")

            with right:

                # Create a pie chart
                fig, ax = plt.subplots()
                ax.pie(predict_count, labels=predict_count.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

                # Display the pie chart in Streamlit
                st.pyplot(fig)


            if st.session_state.predict == 'false': # Evaluasi
                accuracy = accuracy_score(df['sentimen'], df['prediksi'])
                st.write(f"Accuracy: {accuracy * 100:.2f}%")
                st.table(df[['normalized', 'sentimen', 'prediksi']])
            else: # Prediksi
                st.table(df[['normalized', 'prediksi']])


            wc = st.toggle("WordCloud!")

            if wc:
                st.markdown(
                "<div style='font-size:35px;margin-bottom:15px;font-weight:bold;margin-top:50px;'>WordCloud:</div>", 
                unsafe_allow_html=True
                )
                # Filter baris dengan sentimen per kelas
                positif_tokens = df[df['prediksi'] == 'positif']['stemming']
                negatif_tokens = df[df['prediksi'] == 'negatif']['stemming']

                # Menggabungkan semua token menjadi satu list
                all_positif_tokens = [word for tokens in positif_tokens for word in tokens]
                all_negatif_tokens = [word for tokens in negatif_tokens for word in tokens]

                # Menghitung frekuensi kata
                positif_frequencies = Counter(all_positif_tokens) 
                negatif_frequencies = Counter(all_negatif_tokens) 

                # WordCloud untuk sentimen tiap kelas
                wordcloud_positif = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(positif_frequencies)
                wordcloud_negatif = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(negatif_frequencies)

                fig, axes = plt.subplots(1, 2, figsize=(15, 7))

                # Plot WordCloud Positif
                axes[0].imshow(wordcloud_positif, interpolation='bilinear')
                axes[0].set_title('WordCloud Sentimen Positif', fontsize=16)
                axes[0].axis('off')

                # Plot WordCloud Negatif
                axes[1].imshow(wordcloud_negatif, interpolation='bilinear')
                axes[1].set_title('WordCloud Sentimen Negatif', fontsize=16)
                axes[1].axis('off')

                # Tampilkan plot menggunakan Streamlit
                st.pyplot(fig)

            if st.session_state.predict == 'false': # Evaluasi
                cm = st.toggle("Confusion Matrix")
                if cm:
                    tombol = 'true'
                    st.markdown(
                    "<div style='font-size:35px;margin-bottom:15px;font-weight:bold;margin-top:50px;'>Confusion Matrix:</div>", 
                    unsafe_allow_html=True
                    )
                    left, right = st.columns([1,2])
                    asli = df['sentimen']  # Data asli
                    predik = df['prediksi']  # Data prediksi

                    # Membuat confusion matrix
                    cm = confusion_matrix(asli, predik)

                    with left:
                        tn, fp, fn, tp = cm.ravel()
                        st.markdown(
                        "<div style='font-size:25px;margin-bottom:15px;font-weight:bold;'>Informasi Matrix:</div>", 
                        unsafe_allow_html=True
                        )
                        st.write(f"True Negative (TN): {tn}")
                        st.write(f"False Positive (FP): {fp}")
                        st.write(f"False Negative (FN): {fn}")
                        st.write(f"True Positive (TP): {tp}")

                        if tombol == 'true':
                            if st.button("Kembali ke Menu utama"):
                                home_page()
                                st.rerun()



                    with right:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'], ax=ax)
                        ax.set_xlabel('Predicted', fontsize=14)
                        ax.set_ylabel('Actual', fontsize=14)
                        ax.set_title('Confusion Matrix', fontsize=16)

                        # Menampilkan plot di Streamlit
                        st.pyplot(fig)
            if tombol == 'false':
                if st.button("Kembali ke Menu utama"):
                    home_page()
                    st.rerun()


        else: # Satuan (text)
            left, mid, right = st.columns([1,1,1])
            with mid:
                if st.session_state.predict == 'false': # Evaluasi
                    df['sentimen'] = df['sentimen'].replace({1: 'positif', 0: 'negatif'})
                    st.table(df[['content','sentimen','prediksi']])
                
                    if df['sentimen'].loc[0] == df['prediksi'].loc[0]:
                        st.markdown(
                        "<div style='font-size:35px;margin-bottom:15px;font-weight:bold;margin-top:50px;'>Model berhasil menebak!</div>", 
                        unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                        "<div style='font-size:35px;margin-bottom:15px;font-weight:bold;margin-top:50px;'>Model tidak berhasil menebak!</div>", 
                        unsafe_allow_html=True
                        )
                else: # Prediksi
                    st.table(df['2_gram'])
                    
                    st.table(df[['content','prediksi']].transpose())
                    result = df['prediksi'].loc[0]
                    st.markdown(
                    f"<div style='font-size:35px;margin-bottom:15px;font-weight:bold;margin-top:50px;'>Model menebak {result}</div>", 
                    unsafe_allow_html=True
                    )
            
                if st.button("Kembali ke Menu utama"):
                    home_page()
                    st.rerun()
    else: # Evaluasi
        st.title("Hasil testing model")
        df = pd.read_csv("csv/hasil_training.csv")
        df['Sentimen'] = df['Sentimen'].replace({2:0})
        df['Prediksi'] = df['Prediksi'].replace({2:0})
        left, right = st.columns([1,1])

        with left:
            # Membuat confusion matrix
            cm = confusion_matrix(df['Sentimen'], df['Prediksi'])

            # Label untuk axis
            labels = ['Negative', 'Positive']  # Ubah sesuai data Anda

            # Streamlit interface
            st.title("Confusion Matrix")

            # Plot confusion matrix
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_xlabel('Predicted Labels')
            ax.set_ylabel('True Labels')
            ax.set_title('Confusion Matrix')

            # Menampilkan plot di Streamlit
            st.pyplot(fig)
        
        with right:
            st.title('Hasil Data')
            st.dataframe(df[['Content', 'Preprocessing', 'Sentimen', 'Prediksi', 'Category']], hide_index=True)
        
        
        st.title("Word Cloud")
        st.image('asset/wordcloud.png')


        if st.button('Home'):
            home_page()
            st.rerun()

        if st.button('Back'):
            back()
            st.rerun()

# About Page
elif st.session_state.page == 'about':
    # Judul halaman Tentang
    st.title("Tentang Aplikasi Ini")

    # Bagian Deskripsi
    st.write("""
    Aplikasi ini dirancang untuk membantu pengguna mengelola dan menganalisis data dengan lebih efektif. 
    Dengan antarmuka yang mudah digunakan, aplikasi ini menyediakan wawasan 
    dan membantu pengguna dalam memberikan informasi terkait dengan data yang diberikan.
    """)

    # Bagian Tentang Pengembang
    st.header("Tentang Pengembang")
    st.write("""Aplikasi ini dibuat oleh Nicholas. Sebagai salah satu syarat untuk lulus sidang dan seorang penggemar data dengan minat besar dalam membangun 
    alat yang membuat data lebih mudah diakses""")

    # Bagian Fitur
    st.header("Fitur-Fitur")
    st.write("- **Visualisasi Data**: Buat grafik dan diagram untuk mendapatkan wawasan cepat.")
    st.write("- **Analisis Data**: Lakukan analisis dari dasar hingga tingkat lanjut pada data Anda.")
    st.write("- **Manajemen Pengguna**: Kelola pengguna dan izin dengan mudah.")

    # Bagian Kontak
    st.header("Data Diri")
    st.write("- **Nama** : Nicholas")
    st.write("- **NIM**: 32210104")
    st.write("- **Semester**: 7")

    if st.button("Kembali"):
        back()
        st.rerun()