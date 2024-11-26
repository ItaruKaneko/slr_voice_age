# all_age_files_f0_blk.py version
# Trying to change block sample trial
# small data for testing -> change n_data to 125 for full set
# transfer sample to 0.2~0.5 sec chunk

# This version calculate linear regression coef correctly
# calculate f0 waveform of each wav file in 200 samples/sec
# Interporate f0 using scipy.interpolate / interp1d
# apply fft and calculate power spectrumin f0pwr_spectrum
# todo
# construct matrix of 200 samples of power spectrum (1Hz to 200Hz)
# apply linear regression
# versions
#   all_age_files SVC,logistic regresion  - new regression tool

# note
#  f0 sampling rate of f0 : 200 sample/sec using pyworld py.dio
#  f0blk_len : 100 , 0.5sec in 200 sample/sec f0 sampling rate
#  blk_X = [[c0, ... , c199],...] blk_y = [age1, age2 ,...]
#  in pass1==1 calculate size of blk_X, in pass 2, calculate coefs

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
f0blk_len = 100 # 0.5 sec in 200sample/sec
n_data = 50    # number of data - 125 for full set
n_fline= 1000  # number of frequency line to be analyzed
n_decim= 50      # number of decimated samples
n_wave_max =4    # max n_wave

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
# When no-block mode
#  f0_buf: output buffer to store result
#  f0_bp: pointer
#  full f0 contoure will be stored
def process_single_wav_file(pass1,age1,folder1,fn1, blk_X, blk_y, bx, spkr_id1): 
    sz1 = os.path.getsize(folder1 + '/' + fn1)
    n_frame = int(((sz1-44)+16000*2/200)*200/16000/2) # estimation of frame rate
    # n_frame is 200 sample/sec number of f0 samples
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
        if n_frame != len(f0):
            raise ValueError("n_frame does not match")
    else:
        raise ValueError("pass1 should be 1 or 2")
    n_blk = int((n_frame-100)/100)

    # process f0
    if pass1 == 1:
        pass
    elif pass1 == 2:
        #for w_x in range(0, len1-100, 100):  # 0 to len1-100 step 100
        for bx1 in range(0, n_blk):  # 0 to len1-100 step 100
            fftresult=np.fft.fft(f0[bx1*100:bx1*100+100]) # fourie transform
            print('fftresult.shape=', fftresult.shape)
            # when bx = 11,bx * 100 = 110, f0.shape is 495
            blk_X[bx+bx1,0:100]=np.real(fftresult)
            blk_X[bx+bx1,100:200]=np.real(fftresult)
            blk_y[bx+bx1] = age1

    return(n_blk+bx)

# This function is not used in age_files_f0
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


# process single speaker
#  spkr_id1   : speaker id
#  tbl_ix1    : not used currently
# /SPEAKERnn contains multiple wave files
def process_single_speaker(pass1,age1, blk_X, blk_y, bx, spkr_id1):
    # process the speaker specified by spkr_age_record
    bx1 = bx
    f0= np.zeros(buf_len, dtype = float)
    f0[0:buf_len]=100.0   # f0 is buffer of f0, length fixed
    # construct folder path
    spkr_folder = wave_file_folder + '/SPEAKER' + spkr_id1 
    # open wave file
    wav_file_list = os.listdir(spkr_folder)  # get list of wave files
    f0_bp = 0
    n_wave =  len(wav_file_list)
    if n_wave > n_wave_max:
        n_wave = n_wave_max
    print('lentgh of wav_file_list = ', n_wave)
    for wav_id in range(0,n_wave):
        # for wav_fn in wav_file_list:   # process all wav files in single speaker
        wav_fn = wav_file_list[wav_id]
        # for wav_fn in ["010440154.WAV"]:
        # ここで1ファイルの処理の関数を使う
        print(pass1, 1,wav_fn)
        bx1 = process_single_wav_file(pass1,age1, spkr_folder,wav_fn,blk_X, blk_y, bx1, spkr_id1) 
        # returned value is new bx1 after process
    return(bx1)

# process all training data
#  age_file_table : table contains file names of all files
def process_file_list(age_file_table):
    # process pass 1 and pass 2 for all files
    # all_max1
    #  - maximum lenght in all speakers are determined in pass 1 and then kept through pass 2
    all_max1 = 0
    fft_buf1 = np.zeros(buf_len)
    # blk_X, blk_y will be allocated at the end of pass==1
    blk_X = 0
    blk_y = 0

    for pass1 in range(1,3):    # pass1 ==1 and pass1 == 2
        print('pass1 = ', pass1)
        bx = 0
        for table_ix in range(0,n_data):
            spkr_age_record = age_file_table[table_ix]  # get single file contains spkr+id, age
            if pass1 < 3:
                spkr_id1 = spkr_age_record[0]  # speaker number
                age1 = spkr_age_record[1]
                bx = process_single_speaker(pass1,age1, blk_X, blk_y, bx, spkr_id1)
                print('speaker=', spkr_id1, ', age=', age1, ', bx=',bx)
            elif pass1 == 3:
                pass
            else:
                raise ValueError("pass1 error")
        # blk_X, blk_y will be allocated at the end of pass==1
        if pass1==1:
            blk_X = np.zeros((bx,200))
            blk_y = np.zeros(bx)
    return (blk_X,blk_y)


# 現在のフォルダを表示した後指定したフォルダから情報を読み込む
print('corrent directory : ' , os.getcwd())
print('reading ', train_file_list_filename, '...')
train_file_table = read_tab_separated_file(train_file_list_filename)

# データの確認
print(len(train_file_table) , ' data had been raead.')

X_train,y_train = process_file_list(train_file_table)

ix_sort = np.argsort(y_train)
X  = X_train[ix_sort]

f_plt1=plt_save_folder + '/' + 'freqmap.png'
plt.figure(0)
plt.contourf(X)
plt.colorbar()
plt.savefig(f_plt1)   # save plot


test_file_table = read_tab_separated_file(test_file_list_filename)
X_test,y_test = process_file_list(test_file_table)    #<-------------- fix it
# X_test,y_test = process_all_test_data(test_file_table)    #<-------------- fix it


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