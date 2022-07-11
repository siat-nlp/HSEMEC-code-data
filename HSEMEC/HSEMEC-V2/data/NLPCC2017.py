import json
a = open('../NLPCC2017/train_data.json', 'r', encoding='utf8')
a = json.load(a)
o = open('../NLPCC2017/train.txt', 'w', encoding='utf8')
for pair in a:
    src = pair[0][0].strip()
    tgt = pair[1][0].strip()
    src_emo = pair[0][1]
    tgt_emo = pair[1][1]
    o.write(src + '\t' + str(src_emo) + '\t' + tgt + '\t' + str(tgt_emo) + '\n')
o.close()