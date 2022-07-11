import os
import re
import nltk
import random
from data.pytorch_chatbot import cornell_raw_data_to_conv_pairs as cornell_pairs


def compute_vocab_size(clean_questions, clean_answers, min_freq=5, min_count=3):
    s = {}
    for i in clean_questions:
        a = i.split(' ')
        for b in a:
            if b in s.keys():
                s[b] = s[b] + 1
            else:
                s[b] = 1

    for i in clean_answers:
        a = i.split(' ')
        for b in a:
            if b in s.keys():
                s[b] = s[b] + 1
            else:
                s[b] = 1

    before = len(s.items())
    s = {key: value for key, value in s.items() if value >= min_count}
    after = len(s.items())
    word_freq = sorted(s.items(), key=lambda x: x[1], reverse=True)
    print(word_freq)
    num_words = len(word_freq)
    total_freq = 0
    for i in word_freq:
        total_freq += i[1]
    print('num_words:', num_words, ' total_freq:', total_freq)
    print('top:', word_freq[:50], ' total_freq:', total_freq)

    i = 0
    cur_sum = 0
    for j in word_freq:
        cur_sum += j[1]
        i += 1
        if j[1] <= min_freq:
            print('cur_word_freq', j[1])
            print('num_words:', num_words, ' total_freq:', total_freq)
            print('covered_freq:', cur_sum)
            print('vocab_size:', i)
            break


# Doing a first cleaning of the texts
def clean_text(text):
    s = text.lower().strip()
    s = re.sub(r"([.!?])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


# # -------------------------------------------------------------------------------------
#
# data_path = 'cornell'
# # Importing the dataset
# lines = open(os.path.join(data_path, 'movie_lines.txt'), encoding='utf-8',
#              errors='ignore').read().split('\n')
# conversations = open(os.path.join(data_path, 'movie_conversations.txt'), encoding='utf-8',
#                      errors='ignore').read().split('\n')
#
# # Creating a dictiorany that maps each line and its id
# id2line = {}
# for line in lines:
#     _line = line.split(' +++$+++ ')
#     if len(_line) == 5:
#         id2line[_line[0]] = _line[4]
#
# # Creating a list of all the conversations
# conversations_ids = []
# for conversation in conversations[:-1]:
#     _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
#     conversations_ids.append(_conversation.split(','))
#
# # Getting separately the questions and the answers
# questions = []
# answers = []
# for conversations_id in conversations_ids:
#     for i in range(len(conversations_id) - 1):
#         questions.append(id2line[conversations_id[i]])
#         answers.append(id2line[conversations_id[i + 1]])
#
# # Cleaning the questions
# clean_questions = []
# for question in questions:
#     clean_questions.append(clean_text(question))
#
# # Cleaning the answers
# clean_answers = []
# for answer in answers:
#     clean_answers.append(clean_text(answer))
#
# # Filtering out the questions and answers that are too short or too long
# short_questions = []
# short_answers = []
# i = 0
# for question in clean_questions:
#     if 5 <= len(question.split()) <= 50:
#         short_questions.append(question)
#         short_answers.append(clean_answers[i])
#     i += 1
# clean_questions = []
# clean_answers = []
# i = 0
# for answer in short_answers:
#     if 5 <= len(answer.split()) <= 50:
#         clean_answers.append(answer)
#         clean_questions.append(short_questions[i])
#     i += 1
#
# # output
# data = [[clean_questions[i], clean_answers[i]] for i in range(len(clean_questions))]
# random.shuffle(data)
#
# train = open(os.path.join(data_path, 'train.txt'), mode='w')
# valid = open(os.path.join(data_path, 'valid.txt'), mode='w')
# test = open(os.path.join(data_path, 'test.txt'), mode='w')
#
# i = 0
# for pair in data:
#     question = pair[0]
#     answer = pair[1]
#     if i < 10000:
#         test.write(question + '\t' + answer + '\n')
#     elif i < 20000:
#         valid.write(question + '\t' + answer + '\n')
#     else:
#         train.write(question + '\t' + answer + '\n')
#     i += 1
#
# with open(os.path.join(data_path, 'test.txt'), 'r', encoding="utf-8", errors='ignore') as data_f:
#     i = 1
#     for line in data_f:
#         data = line.strip().split('\t')
#         if len(data) != 2:
#             continue
#         src, tgt = data[0], data[1]
#         print('conv:' + str(i) + '\npost:' + src + '\nresponse:' + tgt)
#         i += 1

pairs = cornell_pairs()
random.shuffle(pairs)
data_path = 'cornell'
train = open(os.path.join(data_path, 'train.txt'), mode='w')
valid = open(os.path.join(data_path, 'valid.txt'), mode='w')
test = open(os.path.join(data_path, 'test.txt'), mode='w')
i = 0
for pair in pairs:
    question = pair[0]
    answer = pair[1]
    if i < 4500:
        test.write(question + '\t' + answer + '\n')
    # elif i < 9000:
    #     valid.write(question + '\t' + answer + '\n')
    else:
        train.write(question + '\t' + answer + '\n')
    i += 1
print('num of conversations:', i)
print('num of conversations:', len(pairs))
# question=[]
# answer=[]
# for pair in pairs:
#     question.append(pair[0])
#     answer.append(pair[1])
# compute_vocab_size(question,answer)
