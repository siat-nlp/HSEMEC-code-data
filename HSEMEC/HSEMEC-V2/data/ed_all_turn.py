""" Empathetic Dialogues: *.csv -> *.txt """
import csv
import os
import random
import re

from data.pytorch_chatbot import extractSentencePairs, printLines, loadPrepareData, trimRareWords

_posts = []
_responses = []


def ED_preprocess(data_path, data_type='train', max_hist_len=1):
    csv = open(os.path.join(data_path, f'{data_type}.csv'), mode='r').readlines()
    history = []
    for i in range(1, len(csv)):
        cparts = csv[i - 1].strip().split(',')
        sparts = csv[i].strip().split(',')
        if cparts[0] == sparts[0]:
            present = cparts[5].replace('_comma_', ',')
            history.append(present)
            post = ' '.join(history[-max_hist_len:])
            response = sparts[5].replace('_comma_', ',')

            _posts.append(post)
            _responses.append(response)
        else:
            history = []


if __name__ == '__main__':
    corpus_name = 'Empathetic_Dialogues_All_Turn'
    data_path = './ed_all_turn'
    datafile = os.path.join(data_path, "formatted_pairs.txt")
    delimiter = '\t'
    max_hist_len = 1
    MIN_COUNT = 2
    MAX_LENGTH = 30
    # data_type = 'train'
    for data_type in ['train', 'valid', 'test']:
        ED_preprocess(data_path, data_type, max_hist_len)

    data = [[_posts[i], _responses[i]] for i in range(len(_posts))]

    # Write new csv file
    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in data:
            writer.writerow(pair)

    # Print a sample of lines
    print("\nSample lines from file:")
    printLines(datafile)

    # Load/Assemble voc and pairs
    voc, pairs = loadPrepareData(corpus_name, datafile, MAX_LENGTH)

    # Trim voc and pairs
    pairs = trimRareWords(voc, pairs, MIN_COUNT)

    random.shuffle(pairs)
    # Print some pairs to validate
    print("\npairs:")
    for pair in pairs[:10]:
        print(pair)
    train = open(os.path.join(data_path, 'train.txt'), mode='w')
    # valid = open(os.path.join(data_path, 'valid.txt'), mode='w')
    test = open(os.path.join(data_path, 'test.txt'), mode='w')
    i = 1
    for pair in pairs:
        question = pair[0]
        answer = pair[1]
        # print('conv:' + str(i) + '\npost:' + question + '\nresponse:' + answer)
        if i <= 4500:
            test.write(question + '\t' + answer + '\n')
        # elif i <= 9000:
        #     valid.write(question + '\t' + answer + '\n')
        else:
            train.write(question + '\t' + answer + '\n')
        i += 1
    print('num of conversations:', i)
    print('num of conversations:', len(pairs))
