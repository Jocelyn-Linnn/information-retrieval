# 先切割traing和testing data
# tokenization -> feature selection -> classify
# feature selection只留500 term
# multinomial NB classifier算probability要做smoothing
# 總共13個class，每class有15trainning doc
# 上傳kaggle會用F1-score評比
import os
from nltk.stem import PorterStemmer

#【Step0. 重要變數設定】
training_data = []
testing_data = []
dictionary = set()
feature_number = 500
selected_features = []

# 【Step1. 切割trainging和testing data】
input_folder = 'IRTM'
training_codes = []
with open('training.txt', 'r') as f:
    for line in f:
        codes = line.split()[1:]  # 忽略第一個欄位(class)
        training_codes.extend(codes)

for file_name in os.listdir(input_folder):
    # 去掉副檔名並取得檔案代號
    file_code = os.path.splitext(file_name)[0]
    
    # 將文件完整路徑加入 training_data 或 testing_data
    full_path = os.path.join(input_folder, file_name)
    if file_code in training_codes:
        training_data.append(full_path)
    else:
        testing_data.append(full_path)
# print("Training Data:", training_data)
# print("Testing Data:", testing_data)


# 【Step2. 對training data做tokenization, 找出dictionary】
stopwords_file = './stopwords.txt'
with open(stopwords_file, 'r', encoding='utf-8') as file:
    stop_words = file.read().split() 

ps = PorterStemmer()

for file_path in training_data:
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

    stemmed_tokens = [ps.stem(token) for token in cleaned_tokens]

    dictionary.update(stemmed_tokens)

# print("Dictionary:", dictionary)


# 【Step3. feature selection, Likelihood Ratio, 從dictionary中篩選出前500重要的term】
# 儲存特徵詞與其對應的分數
score_list = []
for t in dictionary:
        score = ComputeFeatureUtility(D, t, c)  # 計算詞彙 t 的特徵分數
        score_list.append((t, score))

sorted_features = sorted(score_list, key=lambda x: x[1], reverse=True)
selected_features = sorted_features[:feature_number]

# 【Step4. train classifier and return model parameters, multinomial NB classifier】