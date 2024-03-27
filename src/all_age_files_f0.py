# all_age_files_f0.py version 3
# This version calculate linear regression coef correctly
# calculate f0 waveform of each wav file in 200 samples/sec
# concatenate all f0 coutour oin each speaker
# Interporate f0 using scipy.interpolate / interp1d
# apply fft and calculate power spectrumin f0pwr_spectrum
# todo
# construct matrix of 200 samples of power spectrum (1Hz to 200Hz)
# apply linear regression
# versions
#   all_age_files SVC,logistic regresion  - new regression tool

import os
import pyworld as pw
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.interpolate  import interp1d
from scipy.io import wavfile
from scipy import signal
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ARDRegression
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score

# ファイル名を指定
train_file_list_filename = "../slr101/speechocean762/train/spk2age"
test_file_list_filename = "../slr101/speechocean762/test/spk2age"
wave_file_folder = "../slr101/speechocean762/WAVE"
plt_save_folder = "../pltsave"
# buf_len = 50000 # size of f0 enverope frequency spectrum
# n_data = 10    # number of data
# n_fline= 100  # number of frequency line to be analyzed
# n_decim= 5      # number of decimated samples

# buf_len = 100000 # size of f0 enverope frequency spectrum
# n_data = 125    # number of data
# n_fline= 1000 # number of frequency line to be analyzed
# n_decim= 10      # number of decimated samples

# buf_len = 100000 # size of f0 enverope frequency spectrum
# n_data = 125    # number of data
# n_fline= 1000  # number of frequency line to be analyzed
# n_decim= 50      # number of decimated samples

buf_len = 100000 # size of f0 enverope frequency spectrum
n_data = 125    # number of data
n_fline= 1000  # number of frequency line to be analyzed
n_decim= 50      # number of decimated samples

# tab 区切りの表をファイルから読み込み、リストのリストで返す
def read_tab_separated_file(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            # タブで区切られたデータを分割
            split_line = line.strip().split('\t')
            data.append(split_line)
    return data


# process single wav file
# calculate f0 trajectory . It will be concatenated to single file for
# each speaker in the caller function.
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

def f0_tailor(f0):
    f0_scaler = -4.5   ## 抑揚を強める場合、プラスに、弱める場合、マイナスにします
    f0_mean = np.mean([x for x in f0 if x!=0])
    f0_std = np.std([x for x in f0 if x!=0])
    f0_modified = []
    for i in f0:
        # print(i)
        if i != 0:
            if f0_scaler > 0:
                if i > f0_mean:
                    single_f0_new = i + f0_std * f0_scaler * ((i-f0_mean)/f0_mean)
                elif i < f0_mean:
                    i_new = i - f0_std * f0_scaler * ((f0_mean-i)/f0_mean)
                    if  i_new > 0:
                        single_f0_new = i_new
                    else:
                        single_f0_new = 1          
                else:
                    single_f0_new = i
            else:
                if i > f0_mean:
                    i_new = i + f0_std * f0_scaler * ((i-f0_mean)/f0_mean)
                    if i_new > f0_mean:
                        single_f0_new = i_new
                    else:
                        single_f0_new = f0_mean          
                elif i < f0_mean:
                    i_new = i - f0_std * f0_scaler * ((f0_mean-i)/f0_mean)
                    if  i_new < f0_mean:
                        single_f0_new = i_new
                    else:
                        single_f0_new = f0_mean
                else:
                    single_f0_new = i
        else:
            single_f0_new = i
        f0_modified.append(single_f0_new)
    return(f0_modified)

# Interpolator is, interporation function for f0 discontinuity
#
class Interpolator():
    def __init__(self, y_array):
        self.x_observed = []
        self.y_observed = []
        self.x_valid = []
        self.y_valid = []
        self.x_to_inter = []
        self.y_to_inter = []
        self.start_flag = 0
        self.end_flag = -1
        self.y_interpolated = []

        for i in range(len(y_array)):
            self.x_observed.append(i)
            self.y_observed.append(y_array[i])

        # get start position (first non-0)
        for i in range(len(y_array)):
            if y_array[i] == 0:
                continue
            else:
                self.start_flag = i 
                break
        # get end position (last non-0)
        y_array_ = y_array[::-1]
        for i in range(len(y_array)):
            if y_array_[i] == 0:
                continue
            else:
                self.end_flag = -i - 1
                break

        self.x_valid = self.x_observed[self.start_flag : len(y_array) + self.end_flag]
        self.y_valid = self.y_observed[self.start_flag : len(y_array) + self.end_flag]


    def pre_interpolate_points(self):
        self.x_to_inter = []
        self.y_to_inter = []
        for i in range(len(self.y_valid)):
            if self.y_valid[i] == 0:
                continue
            else:
                self.x_to_inter.append(self.x_valid[i])
                self.y_to_inter.append(self.y_valid[i])

    def ip_curve(self):
        inter_func = interp1d(self.x_to_inter, self.y_to_inter, kind='slinear')
        # print(type(inter_func(self.x_valid).flatten()))
        self.y_interpolated = [0]*self.start_flag + inter_func(self.x_valid).flatten().tolist()
        self.y_interpolated = self.y_interpolated +[0]*(-self.end_flag)
        return self.y_interpolated

    def clear(self):
        self.x_observed = []
        self.y_observed = []
        self.x_valid = []
        self.y_valid = []
        self.x_to_inter = []
        self.y_to_inter = []
        self.start_flag = 0
        self.end_flag = -1
        self.y_interpolated = []


# end of class Interpolator
        
# interpolate1 interpolatef0 contour

def interpolate1(pre_interf0):
    # interpolation
    # pltw = 2000
    # plt.plot(pre_interf0[0:pltw], linewidth=1, color="blue", label="pre-interpolation")
    # plt.legend(fontsize=10)
    # plt.show()
    a = Interpolator(pre_interf0.tolist())
    a.pre_interpolate_points()
    pro_interf0 = np.array(a.ip_curve())
    a.clear()

    # plt.plot(pro_interf0[0:pltw], linewidth=1, color="red", label="pro-interpolation")
    # plt.legend(fontsize=10)
    # plt.show()
    return(pro_interf0)


def process_single_speaker(pass1,tbl_ix1,spkr_id1):
    # process the speaker specified by spkr_age_record
    f0= np.zeros(buf_len, dtype = float)
    f0[0:buf_len]=100.0
    # construct folder path
    spkr_folder = wave_file_folder + '/SPEAKER' + spkr_id1 
    # open wave file
    wav_file_list = os.listdir(spkr_folder)
    f0_bp = 0
    for wav_fn in wav_file_list:
    # for wav_fn in ["010440154.WAV"]:
        # ここで1ファイルの処理の関数を使う
        print(pass1, tbl_ix1,wav_fn)
        len1 = process_single_wav_file(pass1,spkr_folder,wav_fn,f0,f0_bp)
        if buf_len < f0_bp + len1:
            raise ValueError("buf_len overflow")
        # print(spkr_id, wav_fn, age1, len1, f0_bp)
        f0_bp += len1
    # f0 contains f0 contour for all wave file of speaker
    if pass1 == 2:
        # tailor f0 for continuouscontour
        # f0 = interpolate1(f0)
        # f0 = f0[0:13000]
        f0interpolated = interpolate1(f0)
        f0z_off = f0interpolated - np.mean(f0interpolated)  # subtract mean for zero offset
        # calculate auto correlation
        f0fft1 = np.fft.fft(f0z_off)
        # f0pwr_spectrum = np.abs(f0fft1) ** 2
        # f0niquist = len(f0pwr_spectrum) // 2
        # f0high=f0pwr_spectrum[f0niquist-n_fline+1:f0niquist+1]
        f0niquist = len(f0fft1) // 2
        f0high=f0fft1[f0niquist-n_fline+1:f0niquist+1]
        # f0lsp200 = f0pwr_spectrum[f0niquist-n_fline+1:f0niquist+1]
        if n_decim == 1:
            f0decimated = f0high
        else:
            f0decimated = signal.decimate(f0high, n_decim, n=None, ftype='iir', axis=-1, zero_phase=True)
        return(f0_bp,f0decimated)
    else:
        return(f0_bp,0)


def process_all_train_data(age_file_table):
    # process pass 1 and pass 2 for all files
    # all_max1
    #  - maximum lenght in all speakers are determined in pass 1 and then kept through pass 2
    all_max1 = 0
    fft_buf1 = np.zeros(buf_len)
    f0lsp_tbl = np.zeros((n_data,n_fline//n_decim), dtype = float)
    age_tbl = np.zeros(n_data, dtype=float)
    for pass1 in range(1,3):    # pass1 ==1 and pass1 == 2
        print('pass1 = ', pass1)
        for table_ix in range(0,n_data):
            spkr_age_record = age_file_table[table_ix]
            if pass1 < 3:
                spkr_id1 = spkr_age_record[0]  # speaker number
                age_tbl[table_ix] = int(spkr_age_record[1]) # age of the speaker
                spkr_max1, f0lsp_tbl[table_ix] = process_single_speaker(pass1,table_ix,spkr_id1)
                if all_max1 < spkr_max1:
                    all_max1 = spkr_max1
                print('max_len_in_all_spkr', spkr_max1, all_max1)
            elif pass1 == 3:
                pass
            else:
                raise ValueError("pass1 error")
    x = age_tbl
    m = f0lsp_tbl
    normalized_m = m / m.sum(axis=1, keepdims=True)
    return(normalized_m,x)

def process_all_test_data(age_file_table):


    plt.figure(1)
    plt.figure(2)
    # process pass 1 and pass 2 for all files
    # all_max1
    #  - maximum lenght in all speakers are determined in pass 1 and then kept through pass 2
    all_max1 = 0
    fft_buf1 = np.zeros(buf_len)
    f0lsp_tbl = np.zeros((n_data,n_fline//n_decim), dtype = float)
    age_tbl = np.zeros(n_data, dtype=float)
    for pass1 in range(1,3):    # pass1 ==1 and pass1 == 2
        print('pass1 = ', pass1)
        for table_ix in range(0,n_data):
            spkr_age_record = age_file_table[table_ix]
            if pass1 < 3:
                spkr_id1 = spkr_age_record[0]  # speaker number
                age_tbl[table_ix] = int(spkr_age_record[1]) # age of the speaker
                spkr_max1, f0lsp_tbl[table_ix] = process_single_speaker(pass1,table_ix,spkr_id1)
                if all_max1 < spkr_max1:
                    all_max1 = spkr_max1
                print('max_len_in_all_spkr', spkr_max1, all_max1)
            elif pass1 == 3:
                pass
            else:
                raise ValueError("pass1 error")
    x = age_tbl
    m = f0lsp_tbl
    normalized_m = m / m.sum(axis=1, keepdims=True)
    # calucualate linear regression coefficients
    return(normalized_m,x)

# 現在のフォルダを表示した後指定したフォルダから情報を読み込む
print('corrent directory : ' , os.getcwd())
print('reading ', train_file_list_filename, '...')
train_file_table = read_tab_separated_file(train_file_list_filename)

# データの確認
print(len(train_file_table) , ' data had been raead.')

X_train,y_train = process_all_train_data(train_file_table)

ix_sort = np.argsort(y_train)
X  = X_train[ix_sort]

f_plt1=plt_save_folder + '/' + 'freqmap.png'
plt.figure(0)
plt.contourf(X)
plt.colorbar()
plt.savefig(f_plt1)   # save plot


test_file_table = read_tab_separated_file(test_file_list_filename)
X_test,y_test = process_all_test_data(test_file_table)


# analysis and prediction
for method1 in range(1,7):
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
        model1= MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000)
    elif method1==5:
        method_name = 'Logistic Regression'
        model1= LogisticRegression(max_iter=1000)
    elif method1==6:
        method_name = 'SVM'
        model1= SVC(kernel='linear')
    else:
        raise ValueError("method1 error")

    # not good
    #   method_name = 'Automatic Relevance Determination Regression'
    model1.fit(X_train,y_train)
    y_pred = model1.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(method_name, '  , mse=',mse, ', r2=', r2 )

    plt.figure(method1)
    plt.xlabel('True value')
    plt.ylabel('Predicted value')
    plt.title(method_name)
    plt.scatter(y_test,y_pred)
    f_plt1 = plt_save_folder + '/' + method_name + '.png'
    plt.savefig(f_plt1)
    # MSEとR²スコアを計算

print('end')