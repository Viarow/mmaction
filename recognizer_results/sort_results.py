from __future__ import print_function
import pickle
import numpy as np

results = pickle.load(open("results.pkl",'rb'),encoding='utf-8')


fo = open("accuracy.txt",'w')

class_ind = [x.strip().split()
                 for x in open('/home/jlyu4/mmaction/data/ucf101/annotations/classInd.txt')]
class_mapping = {int(x[0]) - 1 : x[1]  for x in class_ind}

for i in range(0,36):

    result = results[i]
    sorted_score = np.sort(result)
    sorted_index = np.argsort(result)
    top_scores = sorted_score[0][-5:]
    top_index = sorted_index[0][-5:]

    top_scores = np.flip(top_scores)
    top_index = np.flip(top_index)
    score_class = []

    for k in range(0,5):
        class_num = top_index[k]
        score_class.append((top_scores[k], class_mapping[class_num]))

    fo.write(str(score_class))
    fo.write('\n')


fo.close()






