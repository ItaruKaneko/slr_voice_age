# all_age_files_f0.py
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

# ファイル名を指定
age_file_list_filename = "../slr101/speechocean762/test/spk2age"
wave_file_folder = "../slr101/speechocean762/WAVE"
buf_len = 50000 # size of f0 enverope frequency spectrum
n_data = 20    # number of data
n_fline = 200  # number of frequency line to be analyzed

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
        # plt.plot(f0_buf[0:2000])
        # plt.show()
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
        f0lsp200 = f0pwr_spectrum[24801:25001]
        plt.figure(1)
        plt.plot(f0lsp200)
        plt.show()
        return(f0_bp,f0lsp200)
    else:
        return(f0_bp,0)


def process_all_age_file_list(age_file_table,):
    # process pass 1 and pass 2 for all files
    plt.figure(2)
    plt.plot(linewidth=1, color="green", label="F0 contour")
    # all_max1
    #  - maximum lenght in all speakers are determined in pass 1 and then kept through pass 2
    all_max1 = 0
    fft_buf1 = np.zeros(buf_len)
    f0lsp_tbl = np.zeros((n_data,n_fline), dtype = float)
    for pass1 in range(1,2):    # pass1 ==1 and pass1 == 2
        print('pass1 = ', pass1)
        for table_ix in range(1,n_data):
            spkr_age_record = age_file_table[table_ix]
            if pass1 < 3:
                spkr_id1 = spkr_age_record[0]  # speaker number
                age1 = int(spkr_age_record[1]) # age of the speaker
                spkr_max1, f0lsp_tbl[table_ix] = process_single_speaker(pass1,spkr_id1)
                if all_max1 < spkr_max1:
                    all_max1 = spkr_max1
                print('max_len_in_all_spkr', spkr_max1, all_max1)
            elif pass1 == 3:
                pass
            else:
                raise ValueError("pass1 error")

# 現在のフォルダを表示した後指定したフォルダから情報を読み込む
print('corrent directory : ' , os.getcwd())
print('reading ', age_file_list_filename, '...')
age_file_table = read_tab_separated_file(age_file_list_filename)

# データの確認
print(len(age_file_table) , ' data had been raead.')

process_all_age_file_list(age_file_table)

print('done')