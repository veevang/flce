import os

# 指定文件夹路径
folder_paths = [
    # "./data/utility_cache/--topic effective --dataset adult --model AdultMLP --num_parts 8",
    # "./data/utility_cache/--topic effective --dataset bank --model BankMLP",
    "./data/utility_cache/--topic robust --dataset adult --model AdultMLP",
    "./data/utility_cache/--topic robust --dataset bank --model BankMLP",
    "./data/utility_cache/--topic robust --dataset dota2 --model Dota2MLP",
    "./data/utility_cache/--topic robust --dataset tictactoe --model TicTacToeMLP",
]
# 遍历文件夹中的所有文件
for folder_path in folder_paths:
    for filename in os.listdir(folder_path):
        # 获取文件名和文件扩展名a
        name, ext = os.path.splitext(filename)

        if 'data replication --' in name:
            new_name = name.replace('data replication --', 'data replication 1 --')
        elif 'random data generation --' in name:
            new_name = name.replace('random data generation --', 'random data generation 1 --')
        elif 'label flip --' in name:
            new_name = name.replace('label flip --', 'label flip 1 --')
        elif 'low quality data --' in name:
            new_name = name.replace('low quality data --', 'low quality data 1 --')
        # elif "--distribution label skew" in name:
        #     new_name = name.replace('--distribution label skew', 'label skew')
        # else:
        #     raise Exception
        else:
            continue

        # 构造新的文件名
        new_filename = new_name + ext

        # 构造旧的文件路径和新的文件路径
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)
        #
        # old_path = folder_path + '/' + filename
        # new_path = folder_path + '/' + new_filename
        # 重命名文件
        os.rename(old_path, new_path)
