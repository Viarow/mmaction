from __future__ import print_function
import pickle
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Sort the results')
    parser.add_argument('--dataset', type=str, help='dataset type')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    results = pickle.load(open("results.pkl", 'rb'), encoding='utf-8')

    fo = open("accuracy.txt", 'w')

    if args.dataset == 'ucf101':
        class_ind = [x.strip().split()
                     for x in open('/home/jlyu4/mmaction/data/ucf101/annotations/classInd.txt')]

        class_mapping = {int(x[0]) - 1: x[1] for x in class_ind}
    elif args.dataset == 'kinetics400':
        class_ind = [x.strip()
                     for x in open('/home/jlyu4/mmaction/data/kinetics400/annotations/classInd.txt')]
        count = 0
        class_mapping = {}
        for x in class_ind:
            class_mapping.update({count : class_ind[count]})
            count = count + 1
            

    for i in range(0, 10):

        result = results[i]
        sorted_score = np.sort(result)
        sorted_index = np.argsort(result)
        top_scores = sorted_score[0][-5:]
        top_index = sorted_index[0][-5:]

        top_scores = np.flip(top_scores)
        top_index = np.flip(top_index)
        score_class = []

        for k in range(0, 5):
            class_num = top_index[k]
            score_class.append((top_scores[k], class_mapping[class_num]))

        fo.write(str(score_class))
        fo.write('\n')

    fo.close()
    


if __name__ == '__main__':
    main()






