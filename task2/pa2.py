import os
import math
from nltk.stem import PorterStemmer

input_folder = './data/'
output_folder = './output/'
dictionary_file = './dictionary.txt'
stopwords_file = './stopwords.txt'
first_txt_file = "./1.txt"
with open(stopwords_file, 'r', encoding='utf-8') as file:
    stop_words = file.read().split() 


# dictionary(idx, term, df)
dictionary = []
# 儲存每個document的term & tf
document_tokens = []
# 儲存所有term
vocabulary = set()
# 儲存所有document tf-idf vector
document_matrices = {}
# 計算文件總數
N = len([filename for filename in os.listdir(input_folder) if filename.endswith('.txt')])


# 【STEP1.  tokenization】
ps = PorterStemmer()
for filename in os.listdir(input_folder):
    if filename.endswith('.txt'):
        input_file = os.path.join(input_folder, filename)

        with open(input_file, 'r', encoding='utf-8') as file:
            text = file.read()

        # split() 進行空格分割
        tokens = text.split()

        # Lowercasing
        tokens = [token.lower() for token in tokens]

        # Stopword removal
        tokens = [token for token in tokens if token not in stop_words]

        # Remove punctuation
        cleaned_tokens = []
        for token in tokens:
            cleaned_token = ''.join(char for char in token if char.isalnum())
            if cleaned_token:
                cleaned_tokens.append(cleaned_token)

        # Stemming
        stemmed_tokens = [ps.stem(token) for token in cleaned_tokens]

        # 計算tf
        term_frequency = {}
        for token in stemmed_tokens:
            if token in term_frequency:
                term_frequency[token] += 1
            else:
                term_frequency[token] = 1

        # 將當前文件的 token list 加入到 document_tokens 中作為新的一行
        document_tokens.append((filename, term_frequency))

        # 將當前文件中的所有term加入到 vocabulary 中
        vocabulary.update(stemmed_tokens)



# 【STEP2.  計算df/建構dictionary】
# 初始化df
df_dict = {term: 0 for term in vocabulary}

# r計算df
for term in vocabulary:
    for filename, term_frequency in document_tokens:
        if term in term_frequency:
            df_dict[term] += 1

# 建構dictionary
for idx, term in enumerate(sorted(vocabulary), 1):
    dictionary.append([idx, term, df_dict[term]])

# 將dictionary輸出到dictionary.txt檔案
with open(dictionary_file, 'w', encoding='utf-8') as out_file:
    out_file.write(f"{'Index':<10}{'Term':<20}{'DF':<5}\n")
    for row in dictionary:
        out_file.write(f"{row[0]:<10}{row[1]:<20}{row[2]:<5}\n")

print(f"dictionary has been successfully exported to {dictionary_file}")



# 【STEP3.  計算tf-idf / 輸出每個document的tf-idf vector】
# 計算tf-idf
for filename, term_frequency in document_tokens:
    matrix = []
    for idx, term, df in dictionary:
        if term in term_frequency:
            tf = term_frequency[term]
            idf = math.log(N / df) if df != 0 else 0
            weight = tf * idf
            matrix.append((idx, weight))
    document_matrices[filename] = matrix

# 將tf-idf vector輸出到output folder
for filename, matrix in document_matrices.items():
    output_matrix_file = os.path.join(output_folder, f"{filename}_matrix.txt")

    with open(output_matrix_file, 'w', encoding='utf-8') as file:
        file.write(f"{'t_index':<10}{'tf-idf':<10}\n")
        for idx, weight in matrix:
            file.write(f"{idx:<10}{weight:<10.4f}\n")

    if filename == '1.txt':
        with open(first_txt_file, 'w', encoding='utf-8') as file:
            file.write(f"{'t_index':<10}{'tf-idf':<10}\n")
            for idx, weight in matrix:
                file.write(f"{idx:<10}{weight:<10.4f}\n")

print(f"all document's tf-idf unit vector has been successfully exported to {output_folder}")


# 【STEP4.  cosine(Docx, Docy) - return cosine similarity】
def cosine_similarity(tfidf_vector1, tfidf_vector2):
    # 計算內積
    dot_product = 0.0
    for idx1, weight1 in tfidf_vector1:
        for idx2, weight2 in tfidf_vector2:
            if idx1 == idx2:
                dot_product += weight1 * weight2

    # 計算document1 的vector length
    magnitude1 = math.sqrt(sum(weight1 ** 2 for _, weight1 in tfidf_vector1))

    # 計算document2 的vector length
    magnitude2 = math.sqrt(sum(weight2 ** 2 for _, weight2 in tfidf_vector2))

    # 防止除以 0 的情況
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    # 計算 Cosine Similarity
    cosine_sim = dot_product / (magnitude1 * magnitude2)
    return cosine_sim
    print(f"cosine similarity is:  {cosine_sim}")