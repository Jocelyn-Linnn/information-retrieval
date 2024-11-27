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

#【定義找document class函式】
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


# 【定義feature selection, Likelihood Ratio, 給dictionary的每個term一個score】
def ComputeFeatureUtility(documents, terms, classes):
    print(f"Feature score computing")
    n11 = n10 = n01 = n00 = 0

    for file_path in documents:
        tokens = documents[file_path]
        file_class = GetClassFromFile(file_path, training_txt_path)
        
        for c in classes:
            if terms in tokens and file_class == str(c):
                n11 += 1
            elif terms not in tokens and file_class == str(c):
                n10 += 1
            elif terms in tokens and file_class != str(c):
                n01 += 1
            elif terms not in tokens and file_class != str(c):
                n00 += 1
            # print(f"Class: {c}, n11: {n11}, n10: {n10}, n01: {n01}, n00: {n00}")

    N = n11 + n10 + n01 + n00

    p1 = (n11 + n01) / (N)
    p2 = n11 / (n11 + n10)
    p3 = n01 / (n01 + n00)
    print(f"p1: {p1}, p2: {p2}, p3: {p3}")

    # 使用對數計算
    log_numerator = (
        n11 * math.log(p1) +
        n10 * math.log(1 - p1) +
        n01 * math.log(p1) +
        n00 * math.log(1 - p1)
    )

    log_denominator = (
        n11 * math.log(p2) +
        n10 * math.log(1 - p2) +
        n01 * math.log(p3) +
        n00 * math.log(1 - p3)
    )

    # 計算 -2 * log(分子 / 分母)
    score = -2 * (log_numerator - log_denominator)
    print(f"Score for term '{terms}': {score}")
    return score


# 【定義multinomial NB classifier, training phase, return model parameters】
def TrainMultinomialNB(classes, document, vocabulary):
    N = len(document)  # 總文件數
    prior = {}  # P(c)
    condprob = {}  # P(X=t|c)

    # 初始化 condprob 結構
    for t in vocabulary:
        condprob[t] = {}

    # 針對每個類別計算 prior 和 condprob
    for c in classes:
        # 計算 Nc: 該類別的文件數量
        Nc = CountDocsInClass(document, c)

        # 計算 P(c): 該類別的先驗概率
        prior[c] = Nc / N
        print(f"Class {c}: Prior = {prior[c]}")

        # 將屬於該類別的所有文件內容連接成一個 text
        text_c = ConcatenateTextOfAllDocsInClass(document, c)

        # 計算該類別下 vocabulary 中每個詞的出現次數
        T_c = len(text_c)  # 該類別中所有詞的總數
        for t in vocabulary:
            T_ct = text_c.count(t)  # 詞 t 在類別 c 的總出現次數
            # 使用 Laplace Smoothing 計算 P(X=t|c)
            condprob[t][c] = (T_ct + 1) / (T_c + 1)

    return prior, condprob

def CountDocsInClass(document, c):
    count = 0
    for file_path in document:
        file_class = GetClassFromFile(file_path, training_txt_path)
        if file_class == str(c):
            count += 1
    return count

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
def ApplyMultinomialNB(classes, vocabulary, prior, condprob, test_file):
    tokens = tokenization(test_file)
    w = set([t for t in tokens if t in vocabulary])  # 過濾掉不在詞彙表中的詞

    # 初始化每個類別的分數
    score = {}

    # 計算每個類別的分數
    for c in classes:
        # 初始化分數為 log(P(c))
        score[c] = math.log(prior[c])

        # 遍歷詞彙表中的所有詞 V
        for t in w:
                score[c] += math.log(condprob[t][c])

    # 返回分數最高的類別
    predicted_class = max(score, key=score.get)
    return predicted_class



# 【主程式】
#step1. tokenization
for file_path in training_data:
    tokens_result = tokenization(file_path)
    dictionary.update(tokens_result)
    tokenized_training_data[file_path] = tokens_result

for file_path in testing_data:
    tokens_result = tokenization(file_path)
    tokenized_testing_data[file_path] = tokens_result

#step2. feature selection
score_list = []
for t in dictionary:
        score = ComputeFeatureUtility(tokenized_training_data, t, class_set)
        score_list.append((t, score))

sorted_features = sorted(score_list, key=lambda x: x[1], reverse=False)
selected = sorted_features[:feature_number]
selected_features = [t[0] for t in selected]
print(selected_features)

# step3. train multinomial NB classifier
prior, condprob = TrainMultinomialNB(class_set, tokenized_training_data, selected_features)

# Step 4: Predict 並直接輸出成 CSV
results = {}
output_file = "submission.csv"
with open(output_file, 'w', newline='', encoding='utf-8') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(["Id", "Value"])
    
    for test_file in tokenized_testing_data:
        predicted_class = ApplyMultinomialNB(class_set, selected_features, prior, condprob, test_file)
        results[test_file] = predicted_class
        
        file_id = test_file.replace("IRTM\\", "").replace(".txt", "")
        
        csv_writer.writerow([file_id, predicted_class])

print(f"Predictions written to {output_file}")



