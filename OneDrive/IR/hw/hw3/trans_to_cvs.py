import csv

# 讀取 predictions.txt
input_file = "predictions.txt"
output_file = "submission.csv"

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    # CSV writer
    csv_writer = csv.writer(outfile)
    
    # 寫入 header
    csv_writer.writerow(["Id", "Value"])
    
    # 逐行處理 txt 檔案內容
    for line in infile:
        # 分割出檔案名和類別
        file_path, predicted_class = line.strip().split()
        
        # 去掉 IRTM\ 和 .txt，保留檔案代號作為 Id
        file_id = file_path.replace("IRTM\\", "").replace(".txt", "")
        
        # 寫入 CSV 行
        csv_writer.writerow([file_id, predicted_class])

print(f"CSV file '{output_file}' has been created.")
