"""
Usage:
# Split labels for training and testing:
python split_labels.py -i ../../annotations/raccoon_labels.csv -o ../../annotations
"""
import os
import numpy as np
import pandas as pd
import argparse
np.random.seed(1)

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Split labels from CSV")
parser.add_argument("-i", "--inputFile", required=True,
    help="Name of the .csv file (including path) to split", type=str)
parser.add_argument("-o", "--outputPath", required=True,
    help="Name of output path for splitted .csv files", type=str)
args = parser.parse_args()

assert(os.path.isfile(args.inputFile))
assert(os.path.isdir(args.outputPath))

full_labels = pd.read_csv(args.inputFile)

grouped = full_labels.groupby('filename')
grouped_list = [grouped.get_group(x) for x in grouped.groups]

totalSize = len(grouped_list)
trainSize = int(totalSize * .8)
testSize = int(totalSize * .2)

print("[INFO] Total labels = {}".format(len(grouped_list)))
print("[INFO] Train size = {}".format(trainSize))
print("[INFO] Test labels = {}".format(testSize))

train_index = np.random.choice(totalSize, size=trainSize, replace=False)
test_index = np.setdiff1d(list(range(totalSize)), train_index)

train = pd.concat([grouped_list[i] for i in train_index])
test = pd.concat([grouped_list[i] for i in test_index])

train.to_csv(os.path.join(args.outputPath, 'train_labels.csv'), index=None)
test.to_csv(os.path.join(args.outputPath, 'test_labels.csv'), index=None)

print('Successfully generated train and test csv files')
