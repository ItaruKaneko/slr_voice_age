import os

# ファイル名を指定
age_file_list_filename = "../slr101/speechocean762/test/spk2age"
wave_file_folder = "../slr101/speechocean762/WAVE"

# tab 区切りの表をファイルから読み込み、リストのリストで返す
def read_tab_separated_file(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            # タブで区切られたデータを分割
            split_line = line.strip().split('\t')
            data.append(split_line)
    return data

def process_single_wav_file(folder1,fn1, age1):
    sz1 = os.path.getsize(folder1 + '/' + fn1)
    return sz1

def process_all_age_file_list(age_file_table,):
    # age_file_list の全フォルダの全wavファイルを処理するループ
    for spkr_age_record in age_file_table:
        maxlen1 = 0
        # process the speaker specified by spkr_age_record
        spkr_id = spkr_age_record[0]  # speaker number
        age1 = int(spkr_age_record[1]) # age of the speaker
        # construct folder path
        spkr_folder = wave_file_folder + '/SPEAKER' + spkr_id 
        # open wave file
        wav_file_list = os.listdir(spkr_folder)
        for wav_fn in wav_file_list:
            # ここで1ファイルの処理の関数を使う
            len1 = process_single_wav_file(spkr_folder,wav_fn,age1)
            if maxlen1 < len1:
                maxlen1 = len1
            print(spkr_id, wav_fn, age1, len1, maxlen1)
    print('maxlen1 = ', maxlen1)

# 現在のフォルダを表示した後指定したフォルダから情報を読み込む
print('corrent directory : ' , os.getcwd())
print('reading ', age_file_list_filename, '...')
age_file_table = read_tab_separated_file(age_file_list_filename)

# データの確認
print(len(age_file_table) , ' data had been raead.')

process_all_age_file_list(age_file_table)


print('done')