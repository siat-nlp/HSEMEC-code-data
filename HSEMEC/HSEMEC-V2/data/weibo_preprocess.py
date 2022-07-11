import os
import random

def get_data_set(filepath, filename, train_rate, test_rate, valid_rate):
    print('Start reading q-r pairs in the files...')
    pairs = {}
    for file in filename:
        print('Reading q-r pairs in ' + file + '...')
        with open(os.path.join(filepath, file), mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n').split('\t')
                if line[0] in pairs.keys():
                    pairs[line[0]].append(line[1])
                else:
                    pairs[line[0]] = [line[1]]

    # del_que = []
    # for question, answers in pairs.items():
    #     if len(answers) <= 4:
    #         del_que.append(question)
    # for i in del_que:
    #     del pairs[i]

    pair_num = 0
    for question, answers in pairs.items():
        pair_num += len(answers)
    print('Read ' + str(pair_num) + ' q-r pairs')

    print('Start divide dataset into train/test/valid...')
    train = [-1]
    valid = [-1]
    test = [-1]
    questions = list(pairs.keys())
    random.shuffle(questions)
    test_size = int(pair_num * test_rate)
    valid_size = int(pair_num * valid_rate)
    train_size = max(int(pair_num * train_rate), pair_num - test_size - valid_size)
    allocated = 0
    for q in questions:
        for ans in pairs[q]:
            if valid[0] == -1:
                valid.append([q, ans])
                allocated += 1
                if allocated == valid_size:
                    valid[0] = 1
                    allocated = 0
                    print('Valid set q-r pairs size: ' + str(valid_size))
                    break
            elif test[0] == -1:
                test.append([q, ans])
                allocated += 1
                if allocated == test_size:
                    test[0] = 1
                    allocated = 0
                    print('Test set q-r pairs size: ' + str(test_size))
                    break
            else:
                train.append([q, ans])
    print('Train set q-r pairs size: ' + str(len(train)))
    print(train_size)

    print('Start saving dataset...')
    for type in ['valid', 'test', 'train']:
        out = open(os.path.join(filepath, type + '.txt'), mode='w', encoding='utf-8')
        if type == 'valid':
            dataset = valid[1:]
        elif type == 'test':
            dataset = test[1:]
        else:
            dataset = train[1:]
            random.shuffle(dataset)

        for _pair in dataset:
            out.write(_pair[0] + '\t' + _pair[1] + '\n')
        print(type + ' dataset successfully saved.')


if __name__ == '__main__':
    random.seed(5789)
    filepath = 'weibo_utf8'
    filename = ['weibo_src_tgt_utf8.dev', 'weibo_src_tgt_utf8.train']
    get_data_set(filepath, filename, 0.9, 0.05, 0.05)
