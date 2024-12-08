import os
import math
import numpy as np
from nltk.stem import PorterStemmer

stopwords_file = 'stopwords.txt'
data_folder = 'data'

def tokenize_documents(data_folder, stopwords_file):
    with open(stopwords_file, 'r', encoding='utf-8') as file:
        stop_words = set(file.read().split())
    
    tokenized_docs = {}
    dictionary = set()
    ps = PorterStemmer()
    
    for filename in os.listdir(data_folder):
        file_id = int(filename.split('.')[0])
        file_path = os.path.join(data_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        tokens = text.split()
        tokens = [token.lower() for token in tokens]
        tokens = [token for token in tokens if token not in stop_words]
        cleaned_tokens = [
            ''.join(char for char in token if char.isalnum()) for token in tokens
        ]
        cleaned_tokens = [token for token in cleaned_tokens if token]
        stemmed_tokens = [ps.stem(token) for token in cleaned_tokens]
        tokenized_docs[file_id] = stemmed_tokens
        dictionary.update(stemmed_tokens)
    
    return tokenized_docs, dictionary

def compute_tf_df(tokenized_docs, dictionary):
    df_all = {}
    tf_matrix = {}

    for doc, tokens in tokenized_docs.items():

        doc_tf = {}
        for token in tokens:
            doc_tf[token] = doc_tf.get(token, 0) + 1
        tf_matrix[doc] = doc_tf

        for token in dictionary:
            if token in tokens:
                df_all[token] = df_all.get(token, 0) + 1

    return tf_matrix, df_all

def compute_idf(df_all, total_docs):

    idf_all = {}
    
    for token, df in df_all.items():
        idf = math.log(total_docs / df)
        idf_all[token] = idf
    
    return idf_all

def compute_tf_idf(tf_matrix, idf_all):

    tf_idf_matrix = []
    
    for doc, doc_tf in tf_matrix.items():
        doc_tfidf = []

        for token, tf in doc_tf.items():
            idf = idf_all.get(token, 0)
            doc_tfidf.append((token, tf * idf))

        tf_idf_matrix.append(doc_tfidf)

    return tf_idf_matrix


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


def max_similarity(C, I, doc_len):
    max_sim = -1
    doc_i, doc_m = -1, -1
    for i in range(doc_len):
        if I[i] != 1:
            continue
        for m in range(doc_len):
            if I[m] == 1 and i != m:
                if max_sim < C[i][m]:
                    max_sim = C[i][m]
                    doc_i, doc_m = i, m
    return doc_i, doc_m


def write_result(hac_dict, cluster_num):
    with open(f"./{cluster_num}.txt", "w") as f:
        for k, v in hac_dict.items():
            for doc_id in sorted(v):
                f.write(f"{doc_id+1}\n")
            f.write("\n")


if __name__ == "__main__":

    tokenized_docs, dictionary = tokenize_documents(data_folder, stopwords_file)
    print('tokenization done')
    
    total_docs = len(tokenized_docs)
    
    tf_matrix, df_all = compute_tf_df(tokenized_docs, dictionary)
    print('tf and df calculation done')
    
    idf_all = compute_idf(df_all, total_docs)
    print('idf calculation done')
    
    tf_idf_matrix = compute_tf_idf(tf_matrix, idf_all)
    print('tf_idf calculation done')

    C = np.zeros((total_docs, total_docs))
    I = np.ones(total_docs, dtype=int)
    A = []

    for i in range(total_docs):
        for j in range(total_docs):
            C[i][j] = cosine_similarity(tf_idf_matrix[i], tf_idf_matrix[j])
        I[i] = 1
    print('origin similarity calculation done')

    for k in range(total_docs-1):
        i, m = max_similarity(C, I, total_docs)
        A.append((i, m))

        for j in range(total_docs):
            C[i][j] = min(cosine_similarity(tf_idf_matrix[i], tf_idf_matrix[j]), cosine_similarity(tf_idf_matrix[m], tf_idf_matrix[j]))
            C[j][i] = min(cosine_similarity(tf_idf_matrix[j], tf_idf_matrix[i]), cosine_similarity(tf_idf_matrix[j], tf_idf_matrix[m]))
        
        I[m] = 0

    hac_dict = {str(i) : [i] for i in range(total_docs)}
    for doc_i, doc_m in A:
        new_element = hac_dict[str(doc_m)]
        hac_dict.pop(str(doc_m))
        hac_dict[str(doc_i)] += new_element
        if len(hac_dict) == 20:
            write_result(hac_dict, 20)
            print('write to 20.txt success')
        if len(hac_dict) == 13:
            write_result(hac_dict, 13)
            print('write to 13.txt success')
        if len(hac_dict) == 8:
            write_result(hac_dict, 8)
            print('write to 8.txt success')
