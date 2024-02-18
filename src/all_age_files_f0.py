# all_age_files_f0.py
# This version calculate linear regression coef correctly
# calculate f0 waveform of each wav file in 200 samples/sec
# concatenate all f0 coutour in each speaker
# apply fft and calculate power spectrumin f0pwr_spectrum
#
# todo
# construct matrix of 200 samples of power spectrum (1Hz to 200Hz)
# apply linear regression

import os
import pyworld as pw
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.interpolate  import interp1d
from scipy.io import wavfile
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ARDRegression
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ファイル名を指定
train_file_list_filename = "../slr101/speechocean762/train/spk2age"
test_file_list_filename = "../slr101/speechocean762/test/spk2age"
wave_file_folder = "../slr101/speechocean762/WAVE"
buf_len = 50000 # size of f0 enverope frequency spectrum
n_data = 10    # number of data
n_fline= 500  # number of frequency line to be analyzed

# buf_len = 100000 # size of f0 enverope frequency spectrum
# n_data = 125    # number of data
# n_fline= 1000  # number of frequency line to be analyzed

# tab 区切りの表をファイルから読み込み、リストのリストで返す
def read_tab_separated_file(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            # タブで区切られたデータを分割
            split_line = line.strip().split('\t')
            data.append(split_line)
    return data

def process_single_wav_file(pass1,folder1,fn1, f0_buf, f0_bp):
    sz1 = os.path.getsize(folder1 + '/' + fn1)
    n_frame = int(((sz1-44)+16000*2/200)*200/16000/2) # estimation of frame rate
    # assumption sampling rate = 16000, header size = 44
    if pass1 == 1:
        pass
    elif pass1 == 2:
        sr, y = wavfile.read(folder1 + '/' + fn1)     # sr sampling rate, y waveform
        # x = [q/sr for q in np.arange(0, len(y), 1)]
        l = len(y)
        # y1 = np.zeros(len(f0_env_fs),dtype=float)
        # y1[0:l] = y.astype(np.float64)
        y1 = y.astype(np.float64)
        _f0, _time = pw.dio(y1, sr)
        f0 = pw.stonemask(y1, _f0, _time, sr)
        if f0.shape[0] > n_frame:
            raise ValueError("estimated n_frame mismatch")
        sz2 = len(f0)
        f0_buf[f0_bp:f0_bp+sz2] = f0
    else:
        raise ValueError("pass1 should be 1 or 2")
    return(n_frame)

def process_single_speaker(pass1,spkr_id1):
    # process the speaker specified by spkr_age_record
    f0_buf = np.zeros(buf_len)
    # construct folder path
    spkr_folder = wave_file_folder + '/SPEAKER' + spkr_id1 
    # open wave file
    wav_file_list = os.listdir(spkr_folder)
    f0_bp = 0
    for wav_fn in wav_file_list:
        # ここで1ファイルの処理の関数を使う
        len1 = process_single_wav_file(pass1,spkr_folder,wav_fn,f0_buf,f0_bp)
        if buf_len < f0_bp + len1:
            raise ValueError("buf_len overflow")
        # print(spkr_id, wav_fn, age1, len1, f0_bp)
        f0_bp += len1
    if pass1 == 2:
        f0fft1 = np.fft.fft(f0_buf)
        f0pwr_spectrum = np.abs(f0fft1) ** 2
        f0niquist = len(f0pwr_spectrum) // 2
        f0lsp200 = f0pwr_spectrum[f0niquist-n_fline+1:f0niquist+1]
        return(f0_bp,f0lsp200)
    else:
        return(f0_bp,0)


def process_all_train_data(age_file_table):
    # process pass 1 and pass 2 for all files
    # all_max1
    #  - maximum lenght in all speakers are determined in pass 1 and then kept through pass 2
    all_max1 = 0
    fft_buf1 = np.zeros(buf_len)
    f0lsp_tbl = np.zeros((n_data,n_fline), dtype = float)
    age_tbl = np.zeros(n_data, dtype=float)
    for pass1 in range(1,3):    # pass1 ==1 and pass1 == 2
        print('pass1 = ', pass1)
        for table_ix in range(0,n_data):
            spkr_age_record = age_file_table[table_ix]
            if pass1 < 3:
                spkr_id1 = spkr_age_record[0]  # speaker number
                age_tbl[table_ix] = int(spkr_age_record[1]) # age of the speaker
                spkr_max1, f0lsp_tbl[table_ix] = process_single_speaker(pass1,spkr_id1)
                if all_max1 < spkr_max1:
                    all_max1 = spkr_max1
                print('max_len_in_all_spkr', spkr_max1, all_max1)
            elif pass1 == 3:
                pass
            else:
                raise ValueError("pass1 error")
    x = age_tbl
    m = f0lsp_tbl
    return(m,x)

def process_all_test_data(age_file_table):
    # process pass 1 and pass 2 for all files
    # all_max1
    #  - maximum lenght in all speakers are determined in pass 1 and then kept through pass 2
    all_max1 = 0
    fft_buf1 = np.zeros(buf_len)
    f0lsp_tbl = np.zeros((n_data,n_fline), dtype = float)
    age_tbl = np.zeros(n_data, dtype=float)
    for pass1 in range(1,3):    # pass1 ==1 and pass1 == 2
        print('pass1 = ', pass1)
        for table_ix in range(1,n_data):
            spkr_age_record = age_file_table[table_ix]
            if pass1 < 3:
                spkr_id1 = spkr_age_record[0]  # speaker number
                age_tbl[table_ix] = int(spkr_age_record[1]) # age of the speaker
                spkr_max1, f0lsp_tbl[table_ix] = process_single_speaker(pass1,spkr_id1)
                if all_max1 < spkr_max1:
                    all_max1 = spkr_max1
                print('max_len_in_all_spkr', spkr_max1, all_max1)
            elif pass1 == 3:
                pass
            else:
                raise ValueError("pass1 error")
    x = age_tbl
    m = f0lsp_tbl
    # calucualate linear regression coefficients
    return(m,x)

# 現在のフォルダを表示した後指定したフォルダから情報を読み込む
print('corrent directory : ' , os.getcwd())
print('reading ', train_file_list_filename, '...')
train_file_table = read_tab_separated_file(train_file_list_filename)

# データの確認
print(len(train_file_table) , ' data had been raead.')

X_train,y_train = process_all_train_data(train_file_table)

test_file_table = read_tab_separated_file(test_file_list_filename)
X_test,y_test = process_all_test_data(test_file_table)

# analysis and prediction
for method1 in range(1,4):
    if method1==1:
        method_name = 'Linear Regerssion'
        model1= LinearRegression()
    elif method1==2:
        method_name = 'Random Forest Regression'
        model1= RandomForestRegressor()
    elif method1==3:
        method_name = 'Lasso Regression'
        model1= Lasso()
    elif method1==4:
        method_name = 'MLP Regoresso'
        model1= MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000)
    else:
        raise ValueError("method1 error")

    # not good
    #   method_name = 'Automatic Relevance Determination Regression'
    model1.fit(X_train,y_train)
    y_pred = model1.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    plt.figure(method1)
    plt.xlabel('True value')
    plt.ylabel('Predicted value')
    plt.title(method_name)
    plt.scatter(y_test,y_pred)
    plt.show()
    # MSEとR²スコアを計算

print('end')