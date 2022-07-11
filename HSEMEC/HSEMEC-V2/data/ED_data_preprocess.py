""" Empathetic Dialogues: *.csv -> *.txt """
import os
import re
from data.Cornell_Movie_data_preprocess import clean_text, compute_vocab_size

a = []
b = []


def ED_preprocess(data_path, data_type='train', max_hist_len=1):
    csv = open(os.path.join(data_path, f'{data_type}.csv'), mode='r').readlines()
    txt = open(os.path.join(data_path, f'{data_type}.txt'), mode='w')
    history = []
    new_conv = True
    for i in range(1, len(csv)):
        # for i in range(1, 10):
        cparts = csv[i - 1].strip().split(',')
        sparts = csv[i].strip().split(',')
        if cparts[0] == sparts[0]:
            if new_conv:
                present = cparts[5].replace('_comma_', ',')
                history.append(present)
                post = ' '.join(history[-max_hist_len:])
                response = sparts[5].replace('_comma_', ',')

                post = clean_text(post)
                response = clean_text(response)

                a.append(post)
                b.append(response)
                print('conv:' + str(i - 1) + '\npost:' + post + '\nresponse:' + response)
                txt.write(post + '\t' + response + '\n')

                new_conv = False
        else:
            new_conv = True
            history = []


if __name__ == '__main__':
    data_path = './empatheticdialogues'
    max_hist_len = 1
    # data_type = 'train'
    for data_type in ['train','valid','test']:
        ED_preprocess(data_path, data_type, max_hist_len)

    compute_vocab_size(a,b,1)
    # with open(os.path.join(data_path, f'{data_type}.txt'), 'r', encoding="utf8", errors='ignore') as data_f:
    #     i=1
    #     for line in data_f:
    #         data = line.strip().split('\t')
    #         if len(data) != 2:
    #             continue
    #         src, tgt = data[0], data[1]
    #         print('conv:' + str(i) + '\npost:' + src + '\nresponse:' + tgt)
    #         i+=1
