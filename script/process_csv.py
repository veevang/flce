import csv

input_file = '../result/exp_result/final/metric_remove_client_robust_setting_3_0.8.csv'  # 输入CSV文件的名称
output_file = '../result/exp_result/final/metric_remove_client_robust_setting_3_0.8.csv'  # 输出CSV文件的名称

# 读取CSV文件并删除特定行
with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
        open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    reader = csv.DictReader(infile, delimiter=';')  # 指定分隔符为分号
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=';')  # 指定分隔符为分号

    writer.writeheader()
    for row in reader:
        if row['attack_method'] != 'random data generation':
            writer.writerow(row)

print(f'Processed file saved as {output_file}')
