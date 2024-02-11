import os

def read_tab_separated_file(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            # タブで区切られたデータを分割
            split_line = line.strip().split('\t')
            data.append(split_line)
    return data

# ファイル名を指定
age_file_list_filename = "../slr101/speechocean762/test/spk2age"
wave_file_folder = "../slr101/speechocean762/WAVE"

# ファイルを読み込み
print('corrent directory : ' , os.getcwd())
print('reading ', age_file_list_filename, '...')
age_file_list = read_tab_separated_file(age_file_list_filename)

# データの確認
print(len(age_file_list) , ' data had been raead.')

# age_file_list の全要素について処理する

for file_name in age_file_list:
    # ここに各ファイルに対する処理を書く
    print("現在のファイル:", file_name)
    # 例えばファイルを開いて内容を読むなどの処理が入る

print('done')