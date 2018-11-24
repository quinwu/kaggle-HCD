import os
import pandas as pd
import numpy as np

sample_path = '/home/kwu/data/kaggle/HCD/sample_submission.csv'


def average_ensemble(root,avg_path):
    submissions = []
    sample_submission = pd.read_csv(sample_path).set_index('id')

    for fn in os.listdir(root):
        sub = pd.read_csv(os.path.join(root,fn)).set_index('id')
        labels = []
        for i in sample_submission.index.values:
            labels.append(sub.loc[i]['label'])
        sub = pd.DataFrame({'id': sample_submission.index.values, 'label': labels})
        submissions.append(sub.label.values)

    submissions = np.asarray(submissions)
    probs = np.mean(submissions,axis=0)

    ensemble = pd.DataFrame({'id':sample_submission.index.values,'label':probs})
    ensemble.to_csv(avg_path,index=False)

def max_confidences():
    pass

# if __name__ == '__main__':
#     root_path = '/home/kwu/Project/kaggle/HCD/ensemble'
#     average_ensemble(root_path,'/home/kwu/Project/kaggle/HCD/result/avg_ensemble.csv')