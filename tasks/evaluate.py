import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, accuracy_score, \
    classification_report


if __name__ == '__main__':
    pred_folders_path = '../output'
    pred_folders = sorted(list(Path(pred_folders_path).resolve().iterdir()))
    pred_folders = [p for p in pred_folders if p.name.endswith('class')]

    for n_classes in [2, 3]:
        target_folders = [p for p in pred_folders if p.name.endswith(f'{n_classes}-class')]
        test_subs = []
        val_uar_list = []
        for target_folder in target_folders:
            test_subs.append([p for p in target_folder.iterdir() if p.name.startswith('sub') and 'prob' not in p.name][0])
            val_result = pd.read_csv(target_folder / 'val_results.csv')
            val_uar_list.append(val_result.loc[val_result['uar'].argmax, 'uar'])
        test_subs.sort()

        metric_names = ['condition', 'val_uar', 'uar', 'f1', 'accuracy', 'precision', 'sensitivity', 'specificity', 'g-mean']
        metrics_df = pd.DataFrame(np.zeros((len(test_subs), len(metric_names))), columns=metric_names)

        for i, (val_uar, sub_path) in enumerate(zip(val_uar_list, test_subs)):
            metrics = []

            test_labels = pd.read_csv(sub_path.parent / 'test_manifest.csv', header=None).iloc[:, 1]

            pred = pd.read_csv(sub_path, header=None) - 1
            metrics.append(sub_path.name)
            metrics.append(round(val_uar * 100, 1))
            metrics.append(round(balanced_accuracy_score(test_labels, pred) * 100, 1))
            metrics.append(round(f1_score(test_labels, pred, pos_label=0) * 100, 1))
            metrics.append(round(accuracy_score(test_labels, pred) * 100, 1))
            metrics.append(round(precision_score(test_labels, pred) * 100, 1))
            metrics.append(round(classification_report(test_labels, pred, output_dict=True)['0']['recall'] * 100, 1))
            metrics.append(round(classification_report(test_labels, pred, output_dict=True)['1']['recall'] * 100, 1))
            metrics.append(round(geometric_mean_score(test_labels, pred) * 100, 1))
            metrics_df.iloc[i, :] = metrics

            print(confusion_matrix(test_labels, pred, normalize='true'))

        metrics_df.to_csv(f'../output/metrics_{n_classes}-class.csv', sep=',')
