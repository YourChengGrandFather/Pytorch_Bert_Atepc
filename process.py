from config import *
import pandas as pd

def format_sample(file_paths, output_path):
    text = bio = pola = ''
    items = []
    for file_path in file_paths:
        with open(file_path) as f:
            for line in f.readlines():
                # 单独的空行，表示句子间隔
                if line == '\n':
                    items.append({'text': text.strip(), 'bio': bio.strip(), 'pola': pola.strip()})
                    text = bio = pola = ''
                    continue
                # 文本、bio标记、情感极性
                t, b, p = line.split(' ')
                text += t + ' '
                bio += b + ' '
                # 情感极性修正，2表示好评，改为1
                p = str(1) if p.strip() == str(2) else p.strip()
                pola += p + ' '
    df = pd.DataFrame(items)
    df.to_csv(output_path, index=None)

def check_label():
    df = pd.read_csv(TRAIN_FILE_PATH)
    dct = {}
    for index, row in df.iterrows():
        for b, p in zip(row['bio'].split(), row['pola'].split()):
            # 删除异常值
            if b == 'B-ASP' and p == '-1':
                print(index, row)
                df.drop(index=index, inplace=True)
            cnt = dct.get((b,p), 0)
            dct[(b,p)] = cnt+1
    print(dct)
    df.to_csv(TRAIN_FILE_PATH, index=None)


def split_sample():
    file_name = './output/process/atepc.sample.all.csv'
    df = pd.read_csv(file_name)
    df = df.sample(frac=1)
    df.reset_index(inplace=True, drop=True)
    n = len(df)
    df.loc[:int(n*0.8),:].to_csv('./output/process/atepc.sample.train.csv', index=None)
    df.loc[int(n*0.8):,:].to_csv('./output/process/atepc.sample.test.csv', index=None)


def check_label_sample():
    df = pd.read_csv('./output/process/atepc.sample.all.csv')
    dct = {}
    for index, row in df.iterrows():
        for b, p in zip(row['bio'].split(), row['pola'].split()):
            # 删除异常值
            if b == 'B-ASP' and p == '-1':
                print(index, row)
                df.drop(index=index, inplace=True)
            cnt = dct.get((b,p), 0)
            dct[(b,p)] = cnt+1
    print(dct)
    df.to_csv('./output/process/atepc.sample.all.csv', index=None)

if __name__ == '__main__':

    # format_sample([
    #     './input/origin/camera/camera.atepc.train.dat',
    #     './input/origin/car/car.atepc.train.dat',
    #     './input/origin/notebook/notebook.atepc.train.dat',
    #     './input/origin/phone/phone.atepc.train.dat',
    # ], TRAIN_FILE_PATH)

    # format_sample([
    #     './input/origin/camera/camera.atepc.test.dat',
    #     './input/origin/car/car.atepc.test.dat',
    #     './input/origin/notebook/notebook.atepc.test.dat',
    #     './input/origin/phone/phone.atepc.test.dat',
    # ], TEST_FILE_PATH)

    # check_label()

    # format_sample([
    #     './input/origin/camera/camera.atepc.train.dat',
    #     './input/origin/car/car.atepc.train.dat',
    #     './input/origin/notebook/notebook.atepc.train.dat',
    #     './input/origin/phone/phone.atepc.train.dat',
    #     './input/origin/camera/camera.atepc.test.dat',
    #     './input/origin/car/car.atepc.test.dat',
    #     './input/origin/notebook/notebook.atepc.test.dat',
    #     './input/origin/phone/phone.atepc.test.dat',
    # ], './output/process/atepc.sample.all.csv')

    # check_label_sample()

    split_sample()