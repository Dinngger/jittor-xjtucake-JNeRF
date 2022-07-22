
import os
import shutil

dirs = ['Car', 'Coffee', 'Easyship', 'Scar', 'Scarf']
test_back = 'B_'
cnt = 0
for dir in dirs:
    dir_test = '../logs/' + dir + '/' + test_back + 'test'
    for f in os.listdir(dir_test):
        shutil.move(dir_test + '/' + f, '../logs/' + test_back + 'result/' + f)
        cnt += 1
print(f'move {cnt} files')
