# study of speech processing
# Quita 超初心者向けPythonによる音声の解析と再合成〜〜基本周波数F0の調整〜〜
# https://qiita.com/peng_wei/items/e648bd160849dd3b8568

# install pyworld - pip install pyworld
# install soudnfile - pip install soundfile
# install scipy - pip install scipy


import pyworld as pw
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.interpolate  import interp1d
from scipy.io import wavfile


filename = "SPEAKER0001/000010011.WAV"
wave_file_folder = "../slr101/speechocean762/WAVE"

# 波形図を描く
sr, y = wavfile.read(wave_file_folder + '/./' + filename)     # 周波数と振幅値の抽出
x = [q/sr for q in np.arange(0, len(y), 1)]
plt.figure(0)
plt.plot(x,y, color="blue")
plt.show()

# スペクトログラムを描く
plt.figure(1)
plt.specgram(y,Fs=sr)     # スペクトログラムを描く
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()

# 基本周波数（F0）抽出
plt.figure(2)
y = y.astype(np.float64)
_f0, _time = pw.dio(y, sr)
f0 = pw.stonemask(y, _f0, _time, sr)
plt.plot(f0, linewidth=3, color="green", label="F0 contour")
plt.legend(fontsize=10)
plt.show()

# 基本周波数（F0）の調整
plt.figure(3)
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

plt.plot(f0_modified, linewidth=3, color="red", label="F0_modified contour")
plt.legend(fontsize=10)
plt.show()

f0_array = np.asarray(f0_modified)
synthesized = pw.synthesize(f0_array, sp, ap, sr)
synthesized_normalized = synthesized/(np.nanmax(np.abs(synthesized)))
sf.write("./output.wav",synthesized_normalized,16000)


# 基本周波数（F0）の補間
plt.figure(4)
pre_inter_file = "pre-interpolation.wav"
pre_interf0 = f0
for i in range(150, 170):
  pre_interf0[i] = 0

for i in range(210, 230):
  pre_interf0[i] = 0

plt.plot(pre_interf0, linewidth=3, color="blue", label="pre-interpolation_F0 contour")
plt.legend(fontsize=10)
plt.show()

sp = pw.cheaptrick(y, pre_interf0, _time, sr)
ap = pw.d4c(y, pre_interf0, _time, sr)

synthesized = pw.synthesize(pre_interf0, sp, ap, sr)
synthesized_normalized = synthesized/(np.nanmax(np.abs(synthesized)))
sf.write(pre_inter_file,synthesized_normalized,16000)

# F0の補間
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

pro_inter_file = "./pro-interpolation.wav"
a = Interpolator(pre_interf0.tolist())
a.pre_interpolate_points()
pro_interf0 = np.array(a.ip_curve())
a.clear()

plt.plot(pro_interf0, linewidth=3, color="red", label="pro-interpolation_F0 contour")
plt.legend(fontsize=10)
plt.show()

pro_intersp = pw.cheaptrick(y, pro_interf0, _time, sr)
pro_interap = pw.d4c(y, pro_interf0, _time, sr)

synthesized = pw.synthesize(pro_interf0, pro_intersp , pro_interap, sr)
synthesized_normalized = synthesized/(np.nanmax(np.abs(synthesized)))
sf.write(pro_inter_file,synthesized_normalized,16000)