import os
import pandas as pd
import numpy as np

sample_path = '/home/kwu/data/kaggle/HCD/sample_submission.csv'


def average_ensemble(root,avg_path):
    submissions = []
    sample_submission = pd.read_csv(sample_path).set_index('id')

    for fn in os.listdir(root):
        sub = pd.read_csv(os.path.join(root, fn)).set_index('id')
        labels = []
        for i in sample_submission.index.values:
            labels.append(sub.loc[i]['label'])
        sub = pd.DataFrame({'id': sample_submission.index.values, 'label': labels})
        submissions.append(sub.label.values)

    submissions = np.asarray(submissions)
    probs = np.mean(submissions,axis=0)

    ensemble = pd.DataFrame({'id':sample_submission.index.values,'label':probs})
    ensemble.to_csv(avg_path, index=False)

def max_confidences():
    pass

def hard_voting_ensemble(root, ensemble_path):

    files = os.listdir(root)
    submissions = []
    sample_submission = pd.read_csv(sample_path).set_index('id')

    for fn in files:
        sub = pd.read_csv(os.path.join(root, fn)).set_index('id')
        labels = []
        for i in sample_submission.index.values:
            labels.append(sub.loc[i]['label'])
        sub = pd.DataFrame({'id': sample_submission.index.values, 'label': labels})
        submissions.append(sub.label.values)

    submissions = np.asarray(submissions)
    p_voting = np.sum(submissions >= 0.5, axis=0)
    f_voting = len(files) - p_voting
    max_pred = np.max(submissions, axis=0)
    min_pred = np.min(submissions, axis=0)

    preds = []
    for i in range(len(p_voting)):
        if(p_voting[i] >= f_voting[i]):
            preds.append(max_pred[i])
        else:
            preds.append(min_pred[i])

    preds_df = pd.DataFrame({'id':sample_submission.index.values, 'label':preds})
    preds_df.to_csv(ensemble_path, index=False)


def max_ensemble(root, ensemble_path):
    files = os.listdir(root)
    submissions = []
    sample_submission = pd.read_csv(sample_path).set_index('id')

    for fn in files:
        sub = pd.read_csv(os.path.join(root, fn)).set_index('id')
        labels = []
        for i in sample_submission.index.values:
            labels.append(sub.loc[i]['label'])
        sub = pd.DataFrame({'id': sample_submission.index.values, 'label': labels})
        submissions.append(sub.label.values)

    submissions = np.asarray(submissions)
    preds = np.max(submissions, axis=0)

    preds_df = pd.DataFrame({'id':sample_submission.index.values, 'label':preds})
    preds_df.to_csv(ensemble_path, index=False)


if __name__ == '__main__':
    ensemble_path = '/home/kwu/Project/kaggle/HCD/k_fold_csv/resnet50/avg_ensemble.csv'
    root_path = '/home/kwu/Project/kaggle/HCD/k_fold_csv/resnet50'
    average_ensemble(root_path,ensemble_path)
    # hard_voting_ensemble(root_path,ensemble_path)
    # max_ensemble(root_path, ensemble_path)