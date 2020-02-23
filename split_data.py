"""
    3.1 Create train/validation/test splits

    This script will split the data/data.tsv into train/validation/test.tsv files.
"""

import csv
import numpy as np


tmp = []
obj = []
subj = []
count = 0
train_count = 0
val_count = 0
test_count = 0

with open('data/data.tsv') as data:
    reader = csv.reader(data, delimiter='\t')
    for row in reader:
        tmp.append(row)

    for i in range(len(tmp)):
        if count == 0:
            count += 1
            continue
        elif (count > 0) and (tmp[i][1] == '0'):
            obj.append(tmp[i])
        else:
            subj.append(tmp[i])
    length_obj = len(obj)
    length_subj = len(subj)

with open('data/data.tsv') as data, open('data/train.tsv', 'wt') as train, open('data/validation.tsv', 'wt') as val, open('data/test.tsv', 'wt') as test, open('data/overfit.tsv', 'wt') as overfit:
    reader = csv.reader(data, delimiter='\t')
    train_writer = csv.writer(train, delimiter='\t', lineterminator='\n')
    val_writer = csv.writer(val, delimiter='\t', lineterminator='\n')
    test_writer = csv.writer(test, delimiter='\t', lineterminator='\n')
    overfit_writer = csv.writer(overfit, delimiter='\t', lineterminator='\n')

    x = int(0.64 * length_obj)
    y = int(0.8 * length_obj)

    count = 0

    overfit_ind_o = np.random.randint(length_obj, size=(1, 25))[0]
    for ind in overfit_ind_o:
        overfit_writer.writerow(obj[ind])
        overfit_writer.writerow(subj[ind])

    for a in obj:
        if count < x:
            train_writer.writerow(a)
            train_count += 1
        elif (count >= x) and (count < y):
            val_writer.writerow(a)
            val_count += 1
        else:
            test_writer.writerow(a)
            test_count += 1
        count = count + 1

    print("Train data (objectives):", train_count)
    print("Validation data (objectives):", val_count)
    print("Test data (objectives):", test_count)

    x = int(0.64 * length_subj)
    y = int(0.8 * length_subj)
    z = 50
    train_count = 0
    val_count = 0
    test_count = 0
    count = 0
    for a in subj:
        if count < x:
            train_writer.writerow(a)
            train_count += 1
        elif (count >= x) and (count < y):
            val_writer.writerow(a)
            val_count += 1
        else:
            test_writer.writerow(a)
            test_count += 1
        count = count + 1
    print("Train data (subjectives):", train_count)
    print("Validation data (subjectives):", val_count)
    print("Test data (subjectives):", test_count)