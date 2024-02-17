# all_age_files_f0.py improved version
# concatenate all wav files and apply fft <- this has not beem implemented yet

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
max_fulllen = 60000000 # size of f0 enverope frequency spectrum

# tab 区切りの表をファイルから読み込み、リストのリストで返す
def read_tab_separated_file(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            # タブで区切られたデータを分割
            split_line = line.strip().split('\t')
            data.append(split_line)
    return data

def process_single_wav_file(pass1,folder1,fn1, age1,f0_env_fs):
    sz1 = os.path.getsize(folder1 + '/' + fn1)
    n_frame = int(((sz1-44)+16000*2/200)*200/16000/2) # estimation of frame rate
    # assumption sampling rate = 16000, header size = 44
    if pass1 == 1:
        pass
    elif pass1 == 2:
        sr, y = wavfile.read(folder1 + '/' + fn1)     # sr sampling rate, y waveform
        # x = [q/sr for q in np.arange(0, len(y), 1)]
        # plt.figure(0)
        # plt.plot(x,y, color="blue")
        # plt.show()
        l = len(y)
        # y1 = np.zeros(len(f0_env_fs),dtype=float)
        # y1[0:l] = y.astype(np.float64)
        y1 = y.astype(np.float64)
        _f0, _time = pw.dio(y1, sr)
        f0 = pw.stonemask(y1, _f0, _time, sr)
        if f0.shape[0] > n_frame:
            raise ValueError("estimated n_frame mismatch")
        plt.plot(f0)
        # plt.legend(fontsize=10)
        plt.show()
    else:
        raise ValueError("pass1 should be 1 or 2")
    return(n_frame)

def process_all_age_file_list(age_file_table,):
    # process pass 1 and pass 2 for all files
    plt.figure(2)
    plt.plot(linewidth=1, color="green", label="F0 contour")
    fulllen1 = 0
    for pass1 in range(1,3):    # pass1 ==1 and pass1 == 2
        if max_fulllen < fulllen1:
            raise ValueError("fulllen overflow")
        f0_env_fs = np.zeros(fulllen1)
        for spkr_age_record in age_file_table[1:20]:
            # process the speaker specified by spkr_age_record
            spkr_id = spkr_age_record[0]  # speaker number
            age1 = int(spkr_age_record[1]) # age of the speaker
            # construct folder path
            spkr_folder = wave_file_folder + '/SPEAKER' + spkr_id 
            # open wave file
            wav_file_list = os.listdir(spkr_folder)
            for wav_fn in wav_file_list:
                # ここで1ファイルの処理の関数を使う
                len1 = process_single_wav_file(pass1,spkr_folder,wav_fn,age1,f0_env_fs)
                fulllen1 += len1
                print(spkr_id, wav_fn, age1, len1, fulllen1)
        print('maxlen1 = ', fulllen1)

# 現在のフォルダを表示した後指定したフォルダから情報を読み込む
print('corrent directory : ' , os.getcwd())
print('reading ', age_file_list_filename, '...')
age_file_table = read_tab_separated_file(age_file_list_filename)

# データの確認
print(len(age_file_table) , ' data had been raead.')

process_all_age_file_list(age_file_table)

print('done')