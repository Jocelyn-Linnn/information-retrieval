# 先切割traing和testing data
# tokenization -> feature selection -> classify
# feature selection只留500 term
# multinomial NB classifier算probability要做smoothing
# 總共13個class，每class有15trainning doc
# 上傳kaggle會用F1-score評比
import os
import math
from nltk.stem import PorterStemmer

#【重要變數和路徑設定】
document_folder = 'IRTM'
stopwords_file = 'stopwords.txt'
training_txt_path = 'training.txt'
training_data = [] 
testing_data = []
tokenized_training_data = {}
tokenized_testing_data = {}
dictionary = set()
feature_number = 500
selected_features = []
class_set = list(range(1, 14))


# 【切割trainging和testing data】
# 讀training.txt看有哪些檔案是training document
training_codes = []
with open(training_txt_path, 'r') as f:
    for line in f:
        codes = line.split()[1:]  # 忽略第一個欄位(class)
        training_codes.extend(codes)

for file_name in os.listdir(document_folder):
    file_code = os.path.splitext(file_name)[0]  # 去掉副檔名並取得檔案代號

    full_path = os.path.join(document_folder, file_name)  # 將文件完整路徑加入 training_data 或 testing_data
    if file_code in training_codes:
        training_data.append(full_path)
    else:
        testing_data.append(full_path)


# 【定義tokenization函式, 對training data做tokenization, 找出dictionary】
def tokenization(file_path):
    with open(stopwords_file, 'r', encoding='utf-8') as file:
        stop_words = file.read().split() 
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    tokens = text.split()

    tokens = [token.lower() for token in tokens]

    tokens = [token for token in tokens if token not in stop_words]

    cleaned_tokens = []
    for token in tokens:
        cleaned_token = ''.join(char for char in token if char.isalnum())
        if cleaned_token:
            cleaned_tokens.append(cleaned_token)
            
    ps = PorterStemmer()
    stemmed_tokens = [ps.stem(token) for token in cleaned_tokens]
    return stemmed_tokens

for file_path in training_data:
    tokens_result = tokenization(file_path)
    dictionary.update(tokens_result)
    tokenized_training_data[file_path] = tokens_result


print(tokenized_training_data)

