# 先切割traing和testing data
# tokenization -> feature selection -> classify
# feature selection只留500 term
# multinomial NB classifier算probability要做smoothing
# 總共13個class，每class有15trainning doc
# 上傳kaggle會用F1-score評比
import os
import math
from nltk.stem import PorterStemmer

#【Step1. 重要變數設定】
training_data = []
testing_data = []
tokenized_training_data = {}
dictionary = set()
feature_number = 500
selected_features = []
training_txt_path = "C:\Users\jocel\OneDrive\IR\hw\hw3\training.txt"
class_set = list(range(1, 14))


# 【Step2. 切割trainging和testing data】
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

#【Step3. 定義找class函式】
def GetClassFromFile(file_path, training_txt_path):
    file_name = os.path.splitext(os.path.basename(file_path))[0]  # 提取檔案代號 (不含副檔名)
    with open(training_txt_path, 'r') as file:
        for line in file:
            parts = line.strip().split()  # 將每行拆分成列表
            class_label = parts[0]  # 第一個欄位是類別
            file_numbers = parts[1:]  # 後續欄位是該類別的檔案號碼
            
            if file_name in file_numbers:  # 如果檔案號碼在該類別的檔案列表中
                return class_label  # 返回該類別
    
    # 如果找不到，返回 None 或報錯
    raise ValueError(f"File {file_name} not found in {training_txt_path}")


# 【Step2. 對training data做tokenization, 找出dictionary】
stopwords_file = './stopwords.txt'
with open(stopwords_file, 'r', encoding='utf-8') as file:
    stop_words = file.read().split() 

ps = PorterStemmer()

def tokenization(file_path):
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
    return stemmed_tokens

for file_path in training_data:
    stemmed_tokens_result = tokenization(file_path)
    dictionary.update(stemmed_tokens_result)
    tokenized_training_data[file_path] = stemmed_tokens_result

# print("Dictionary:", dictionary)


# 【Step3. feature selection, Likelihood Ratio, 從dictionary中篩選出前500重要的term】
# 儲存特徵詞與其對應的分數
def ComputeFeatureUtility(documents, terms, classes):
    n11 = n10 = n01 = n00 = 0
    N = len(training_data)

    for file_path in documents:
        tokens = tokenized_training_data[file_path]
        file_class = GetClassFromFile(file_path, training_txt_path)
        
        for c in classes:
            if terms in tokens and file_class == c:
                n11 += 1
            elif t not in tokens and file_class == c:
                n10 += 1
            elif t in tokens and file_class != c:
                n01 += 1
            elif t not in tokens and file_class != c:
                n00 += 1

        p1 = (n11 + n01) / N
        p2 = (n11 + n10) / (n11 + n10) 
        p3 = (n01 + n00) / (n01 + n00)

        # 計算分子
        numerator = (
            (p1 ** n11) * ((1 - p1) ** n10) *
            (p1 ** n01) * ((1 - p1) ** n00)
        )

        # 計算分母
        denominator = (
            (p2 ** n11) * ((1 - p2) ** n10) *
            (p3 ** n01) * ((1 - p3) ** n00)
        )

        # 避免分母為零
        if denominator == 0:
            return 0

        # 計算 -2 * log(分子 / 分母)
        score = -2 * math.log(numerator / denominator + 1e-10)  # 加入平滑項避免 log(0)
        return score
 

score_list = []
for t in dictionary:
        c = list(range(1, 14))
        score = ComputeFeatureUtility(training_data, t, c)  # 計算詞彙 t 的特徵分數
        score_list.append((t, score))

sorted_features = sorted(score_list, key=lambda x: x[1], reverse=True)
selected_features = sorted_features[:feature_number]


# 【Step4. 定義multinomial NB classifier, training phase, return model parameters】
def TrainMultinomialNB(classes, training_data, vocabulary):
    N = len(training_data)  # 總文件數
    prior = {}  # P(c)
    condprob = {}  # P(X=t|c)

    # 初始化 condprob 結構
    for t in vocabulary:
        condprob[t] = {}

    # 針對每個類別計算 prior 和 condprob
    for c in classes:
        # 計算 Nc: 該類別的文件數量
        Nc = CountDocsInClass(training_data, c)

        # 計算 P(c): 該類別的先驗概率
        prior[c] = Nc / N

        # 將屬於該類別的所有文件內容連接成一個 text，同時做tokenization
        text_c = ConcatenateTextOfAllDocsInClass(training_data, c)

        # 計算該類別下 vocabulary 中每個詞的出現次數
        T_c = len(text_c)  # 該類別中所有詞的總數
        for t in vocabulary:
            T_ct = text_c.count(t)  # 詞 t 在類別 c 的總出現次數
            # 使用 Laplace Smoothing 計算 P(X=t|c)
            condprob[t][c] = (T_ct + 1) / (T_c + len(vocabulary))

    return prior, condprob

def CountDocsInClass(training_data, c):
    count = 0
    for file_path in training_data:
        file_class = GetClassFromFile(file_path, 'training.txt')
        if file_class == str(c):
            count += 1
    return count

def ConcatenateTextOfAllDocsInClass(training_data, c):
    concatenated_text = []
    for file_path in training_data:
        # 確認文件所屬類別
        file_class = GetClassFromFile(file_path, 'training.txt')
        if file_class == str(c):  # 類別匹配
            # 對文件進行 tokenization，並將結果合併到 concatenated_text 中
            tokens = tokenized_training_data[file_path]
            concatenated_text.extend(tokens)
    return concatenated_text


# 【Step4. 定義multinomial NB classifier, testing phase, return document class】
def ApplyMultinomialNB(classes, vocabulary, prior, condprob, test_file):
    # 將測試文件進行 tokenization，僅保留詞彙表中的詞
    tokens = tokenization(test_file)
    W = [t for t in tokens if t in vocabulary]  # 過濾掉不在詞彙表中的詞

    # 初始化每個類別的分數
    score = {}

    # 計算每個類別的分數
    for c in classes:
        # 初始化分數為 log(P(c))
        score[c] = math.log(prior[c])

        # 對 W 中每個詞 t 加上 log(P(X=t|c))
        for t in W:
            if t in condprob and c in condprob[t]:
                score[c] += math.log(condprob[t][c])

    # 返回分數最高的類別
    predicted_class = max(score, key=score.get)
    return predicted_class


# 【Step4. multinomial NB classifier主程式, 呼叫前面的函式】
prior, condprob = TrainMultinomialNB(class_set, training_data, dictionary)
# 輸出結果檢查
print("Prior probabilities (P(c)):")
for c in class_set:
    print(f"Class {c}: {prior[c]}")

print("\nConditional probabilities (P(X=t|c)) for a few terms:")
sample_terms = list(dictionary)[:5]  # 選擇字典中前 5 個詞作檢查
for t in sample_terms:
    print(f"Term '{t}':")
    for c in class_set:
        print(f"  Class {c}: {condprob[t][c]}")

results = {}
output_file = "predictions.txt"
with open(output_file, 'w', encoding='utf-8') as file:
    for test_file in testing_data:
        # 對每個測試文件進行分類
        predicted_class = ApplyMultinomialNB(class_set, selected_features, prior, condprob, test_file)
        results[test_file] = predicted_class
        # 寫入檔案，每行格式：測試文件 預測類別
        file.write(f"{test_file} {predicted_class}\n")

print(f"Predictions written to {output_file}")


