# This code loads the MOROCO data set into memory. It is provided for convenience.
# The data set can be downloaded from <https://github.com/butnaruandrei/MOROCO>.
#
# Copyright (C) 2018  Andrei M. Butnaru, Radu Tudor Ionescu
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or any
# later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Assume the data set is in the below subfolder
import os

dirname = os.path.dirname(__file__)
inputDataPrefix = r"MOROCO\preprocessed"
filename = os.path.join(dirname, inputDataPrefix)


# Loads the samples in the train, validation, or test set
def load_moroco_data_samples(subset_name):
    input_samples_file_path = (inputDataPrefix + r"\%s\samples.txt") % subset_name
    input_dialect_labels_file_path = (inputDataPrefix + r"\%s\dialect_labels.txt") % subset_name
    input_category_labels_file_path = (inputDataPrefix + r"\%s\category_labels.txt") % subset_name

    ids = []
    samples = []
    dialect_labels = []
    category_labels = []

    # Loading the data samples
    input_samples_file = open(input_samples_file_path, 'r', encoding='utf-8')
    sample_rows = input_samples_file.readlines()
    input_samples_file.close()

    for row in sample_rows:
        components = row.split("\t")
        ids += [components[0]]
        samples += [" ".join(components[1:])]

    # Loading the dialect labels
    input_dialect_labels_file = open(input_dialect_labels_file_path, 'r', encoding='utf-8')
    dialect_rows = input_dialect_labels_file.readlines()
    input_dialect_labels_file.close()

    for row in dialect_rows:
        components = row.split("\t")
        dialect_labels += [int(components[1])]

    # Loading the category labels
    input_category_labels_file = open(input_category_labels_file_path, 'r', encoding='utf-8')
    category_rows = input_category_labels_file.readlines()
    input_category_labels_file.close()

    for row in category_rows:
        components = row.split("\t")
        category_labels += [int(components[1])]

    # ids[i] is the ID of the sample samples[i] with the dialect label dialect_labels[i] and the category label
    # category_labels[i]
    return ids, samples, dialect_labels, category_labels


# Loads the data set
def load_moroco_data_set():
    train_ids, train_samples, train_dialect_labels, train_category_labels = load_moroco_data_samples("train")
    train_list = [train_ids, train_samples, train_dialect_labels, train_category_labels]
    print("Loaded %d training samples..." % len(train_samples))

    validation_ids, validation_samples, validation_dialect_labels, validation_category_labels = \
        load_moroco_data_samples("validation")
    print("Loaded %d validation samples..." % len(validation_samples))
    validation_list = [validation_ids, validation_samples, validation_dialect_labels, validation_category_labels]

    test_ids, test_samples, test_dialect_labels, test_category_labels = load_moroco_data_samples("test")
    print("Loaded %d test samples..." % len(test_samples))
    test_list = [test_ids, validation_samples, validation_dialect_labels, validation_category_labels]
    return train_list, validation_list, test_list

    # The MOROCO data set is now loaded in the memory.
    # Implement your own code to train and evaluation your own model from this point on.
    # Perhaps you want to return the variables or transform them into your preferred format first...
