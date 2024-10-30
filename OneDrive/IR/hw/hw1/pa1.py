from nltk.stem import PorterStemmer


input_file = './1.txt'

# 讀取文件內容
with open(input_file, 'r', encoding='utf-8') as file:
    text = file.read()

# Tokenization - 使用 split() 函數進行基於空格的分割
tokens = text.split()

# 去掉標點符號
cleaned_tokens = []
for token in tokens:
    cleaned_token = ''.join(char for char in token if char.isalnum())
    if cleaned_token:
        cleaned_tokens.append(cleaned_token)

# Lowercasing
cleaned_tokens = [token.lower() for token in cleaned_tokens]

# Stopword removal
stopwords_file = './stopwords.txt'
with open(stopwords_file, 'r', encoding='utf-8') as file:
    stop_words = file.read()
filtered_tokens = [token for token in cleaned_tokens if token not in stop_words]


# Stemming
ps = PorterStemmer()
stemmed_tokens = [ps.stem(token) for token in filtered_tokens]


# 結果寫入
output_file = './result.txt'
with open(output_file, 'w', encoding='utf-8') as file:
    file.write(' '.join(stemmed_tokens))

print(f"處理完成，結果已儲存至 {output_file}")
