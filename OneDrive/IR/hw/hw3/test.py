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

print(training_codes)