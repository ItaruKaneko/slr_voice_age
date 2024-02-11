def read_tab_separated_file(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            # タブで区切られたデータを分割
            split_line = line.strip().split('\t')
            data.append(split_line)
    return data

# ファイル名を指定
filename = "C:\Users\itaru\OneDrive\prj-07\expr2024\2024openslr_speech\slr101\speechocean762\test\spk2age"

# ファイルを読み込み
data_array = read_tab_separated_file(filename)

# データの確認
print(data_array)