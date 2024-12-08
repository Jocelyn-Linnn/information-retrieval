# 先切割traing和testing data
# tokenization -> feature selection -> classify
# feature selection只留500 term
# multinomial NB classifier算probability要做smoothing
# 總共13個class，每class有15trainning doc
# 上傳kaggle會用F1-score評比
import os
import math
import csv
from nltk.stem import PorterStemmer

#【重要變數和路徑設定】
document_folder = 'Data'
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

# 【定義tokenization函式】
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

# 【定義找document class函式】
def GetClassFromFile(file_path, training_txt_path):
    file_name = os.path.splitext(os.path.basename(file_path))[0]  # 提取檔案代號 (不含副檔名)
    with open(training_txt_path, 'r') as file:
        for line in file:
            parts = line.strip().split()  # 將每行拆分成列表
            class_label = parts[0]  # 第一個欄位是類別
            file_numbers = parts[1:]  # 後續欄位是該類別的檔案號碼
            
            if file_name in file_numbers: 
                return class_label 

    # 如果找不到，返回 None 或報錯
    raise ValueError(f"File {file_name} not found in {training_txt_path}")


# 【定義feature selection, 給dictionary的每個term一個score】
def ComputeFeatureUtility(documents, terms, classes):
    n11 = n10 = n01 = n00 = N = 0

    for c in classes:            
        for file_path in documents:
            tokens = documents[file_path]
            file_class = GetClassFromFile(file_path, training_txt_path)

            if terms in tokens and file_class == str(c):
                n11 += 1
            elif terms not in tokens and file_class == str(c):
                n10 += 1
            elif terms in tokens and file_class != str(c):
                n01 += 1
            elif terms not in tokens and file_class != str(c):
                n00 += 1

    N = n11 + n10 + n01 + n00
    print(n11, n10, n01, n00)
    E11 = N * (n11+n01)/N * (n11+n10)/N
    E10 = N * (n10+n00)/N * (n10+n11)/N
    E01 = N * (n01+n11)/N * (n01+n00)/N
    E00 = N * (n00+n10)/N * (n00+n01)/N
    print(E11, E10, E01, E00)

    score = (n11-E11)**2/E11 + (n10-E10)**2/E10 + (n01-E01)**2/E01 + (n00-E00)**2/E00
    print(f"Score for term '{terms}': {score}")
    return score


# 【定義multinomial NB classifier, training phase, return model parameters】
def TrainMultinomialNB(classes, document, vocabulary):
    prior = {}  # P(c)
    condprob = {}  # P(X=t|c)

    # 初始化 condprob 結構
    for t in vocabulary:
        condprob[t] = {}

    # 針對每個類別計算 prior 和 condprob
    for c in classes:
        prior[c] = 15 / 195

        # 將屬於該類別的所有文件內容連接成一個 text
        text_c = ConcatenateTextOfAllDocsInClass(document, c)

        # 計算該類別下 vocabulary 中每個詞的出現次數
        T_c = len(text_c) 
        for t in vocabulary:
            T_ct = text_c.count(t)
            # 使用 Laplace Smoothing 計算 P(X=t|c)
            condprob[t][c] = (T_ct + 1) / (T_c + 1)
            print(f"{t} condprob: {condprob[t][c]}")

    return prior, condprob

def ConcatenateTextOfAllDocsInClass(document, c):
    concatenated_text = []
    for file_path in document:
        # 確認文件所屬類別
        file_class = GetClassFromFile(file_path, training_txt_path)
        if file_class == str(c):  # 類別匹配
            tokens = document[file_path]
            concatenated_text.extend(tokens)
    return concatenated_text


# 【定義multinomial NB classifier, testing phase, return document class】
def ApplyMultinomialNB(classes, vocabulary, prior, condprob, documents):
    w = [t for t in documents if t in vocabulary]  # 過濾掉不在詞彙表中的詞
    print(w)

    # 初始化每個類別的分數
    score = {}

    # 計算每個類別的分數
    for c in classes:
        score[c] = math.log(prior[c])

        # 遍歷詞彙表中的所有詞 V
        for t in w:
                print(f"adding: {condprob[t][c]}")
                score[c] += math.log(condprob[t][c])
        
        print(f"class {c} score: {score[c]}")

    # 返回分數最高的類別
    predicted_class = max(score, key=score.get)
    return predicted_class


# 【主程式】
# step0. 切割trainging和testing data
# 讀training.txt看有哪些檔案是training document
print("spliting data")
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
print("done")

#step1. tokenization & get dictionary
print("doing tokenization")
for file_path in training_data:
    tokens_result = tokenization(file_path)
    tokenized_training_data[file_path] = tokens_result
    dictionary.update(tokens_result)

for file_path in testing_data:
    tokens_result = tokenization(file_path)
    tokenized_testing_data[file_path] = tokens_result
print("done")

#step2. feature selection
print("doing feature selection")
score_list = []
round = 0
length = len(dictionary)
for t in dictionary:
        score = ComputeFeatureUtility(tokenized_training_data, t, class_set)
        score_list.append((t, score))
        round += 1
        print(f"{round} / {length}")

sorted_features = sorted(score_list, key=lambda x: x[1], reverse=True)
selected = sorted_features[:feature_number]
selected_features = [t[0] for t in selected]
print(selected_features)
print("done")

# step3. train multinomial NB classifier
print("training model")
prior, condprob = TrainMultinomialNB(class_set, tokenized_training_data, selected_features)
print("done")

# Step 4: Predict 並直接輸出成 CSV
print("predicting")
results = {}
output_file = "submission.csv"
with open(output_file, 'w', newline='', encoding='utf-8') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(["Id", "Value"])
    
    for test_file in tokenized_testing_data:
        print(f"predicting doc {test_file}")
        predicted_class = ApplyMultinomialNB(class_set, selected_features, prior, condprob, test_file)
        results[test_file] = predicted_class
        
        file_id = test_file.replace("Data\\", "").replace(".txt", "")
        
        csv_writer.writerow([file_id, predicted_class])

print(f"Predictions written to {output_file}")



