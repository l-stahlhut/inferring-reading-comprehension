import numpy as np
import os

#
# nan values? ---------------------------------------------------------------------------------------------------------
# X_train = np.load("nn/indico_splits/data_s1_rm1_lf1_pos_cont/book/X_train_book_0.npy")
# X_test = np.load("nn/indico_splits/data_s1_rm1_lf1_pos_cont/book/y_train_book_0.npy")
# y_train = np.load("nn/indico_splits/data_s1_rm1_lf1/book/X_test_book_0.npy")
# y_test = np.load("nn/indico_splits/data_s1_rm1_lf1/book/y_test_book_0.npy")
X_train = np.load("nn/sbsat_splits/data_s1_rm0_lf0/book/X_train_book_0.npy")
y_train = np.load("nn/sbsat_splits/data_s1_rm0_lf0/book/y_train_book_0.npy")
X_test = np.load("nn/sbsat_splits/data_s1_rm0_lf0/book/X_test_book_0.npy")
y_test = np.load("nn/sbsat_splits/data_s1_rm0_lf0/book/y_test_book_0.npy")
#print(X_train)
print(X_train.shape) # book setting: test is 25% of total dataset (4-fold cross validation)
print(X_test.shape)
#print(np.isnan(X_train).any())
#print(np.isnan(y_train).any())
#print(np.isnan(X_test).any())
#print(np.isnan(y_test).any())

#
# # ---------------------------------------------------------------------------------------------------------------------
# # ------------------------------------------------------------------
# # SBSAT -----------------------------------------------------------
# # ------------------------------------------------------------------
# print("SBSAT: ")
# # All features ------------------------------------------------------
# print("All features")
# dataset_path = "nn/sbsat_splits/data_s1_rm1_lf1"
# criterions = ["book", "book-page", "subj"]
# for criterion in criterions:
#     print("\nCriterion: ", criterion)
#     print("X_train: ", np.load(os.path.join(dataset_path, criterion, 'X_train_' + criterion + '_0.npy')).shape)
#     print("y_train: ", np.load(os.path.join(dataset_path, criterion, 'y_train_' + criterion + '_0.npy')).shape)
#     print("X_test: ", np.load(os.path.join(dataset_path, criterion, 'X_test_' + criterion + '_0.npy')).shape)
#     print("y_test: ", np.load(os.path.join(dataset_path, criterion, 'y_test_' + criterion + '_0.npy')).shape)
#
# # Some features ------------------------------------------------------
# print("\nSome features: ")
# dataset_path = "nn/sbsat_splits/data_s1_rm1_lf1_pos_cont"
# criterions = ["book", "book-page", "subj"]
# for criterion in criterions:
#     print("\nCriterion: ", criterion)
#     print("X_train: ", np.load(os.path.join(dataset_path, criterion, 'X_train_' + criterion + '_0.npy')).shape)
#     print("y_train: ", np.load(os.path.join(dataset_path, criterion, 'y_train_' + criterion + '_0.npy')).shape)
#     print("X_test: ", np.load(os.path.join(dataset_path, criterion, 'X_test_' + criterion + '_0.npy')).shape)
#     print("y_test: ", np.load(os.path.join(dataset_path, criterion, 'y_test_' + criterion + '_0.npy')).shape)
#
# # All features ------------------------------------------------------
# print("\nNo features: ")
# dataset_path = "nn/sbsat_splits/data_s1_rm1_lf0"
# criterions = ["book", "book-page", "subj"]
# for criterion in criterions:
#     print("\nCriterion: ", criterion)
#     print("X_train: ", np.load(os.path.join(dataset_path, criterion, 'X_train_' + criterion + '_0.npy')).shape)
#     print("y_train: ", np.load(os.path.join(dataset_path, criterion, 'y_train_' + criterion + '_0.npy')).shape)
#     print("X_test: ", np.load(os.path.join(dataset_path, criterion, 'X_test_' + criterion + '_0.npy')).shape)
#     print("y_test: ", np.load(os.path.join(dataset_path, criterion, 'y_test_' + criterion + '_0.npy')).shape)
#
#
# # ------------------------------------------------------------------
# # InDiCo -----------------------------------------------------------
# # ------------------------------------------------------------------
# print("InDiCo")
# # All features ------------------------------------------------------
# print("All features")
# dataset_path = "nn/indico_splits/data_s1_rm1_lf1"
# criterions = ["book", "book-page", "subj"]
# for criterion in criterions:
#     print("\nCriterion: ", criterion)
#     print("X_train: ", np.load(os.path.join(dataset_path, criterion, 'X_train_' + criterion + '_0.npy')).shape)
#     print("y_train: ", np.load(os.path.join(dataset_path, criterion, 'y_train_' + criterion + '_0.npy')).shape)
#     print("X_test: ", np.load(os.path.join(dataset_path, criterion, 'X_test_' + criterion + '_0.npy')).shape)
#     print("y_test: ", np.load(os.path.join(dataset_path, criterion, 'y_test_' + criterion + '_0.npy')).shape)
#
# # Some linguistic features ------------------------------------------------------
# print("\nSome features: ")
# dataset_path = "nn/indico_splits/data_s1_rm1_lf1_pos_cont"
# criterions = ["book", "book-page", "subj"]
# for criterion in criterions:
#     print("\nCriterion: ", criterion)
#     print("X_train: ", np.load(os.path.join(dataset_path, criterion, 'X_train_' + criterion + '_0.npy')).shape)
#     print("y_train: ", np.load(os.path.join(dataset_path, criterion, 'y_train_' + criterion + '_0.npy')).shape)
#     print("X_test: ", np.load(os.path.join(dataset_path, criterion, 'X_test_' + criterion + '_0.npy')).shape)
#     print("y_test: ", np.load(os.path.join(dataset_path, criterion, 'y_test_' + criterion + '_0.npy')).shape)
#
# # No linguistic features ------------------------------------------------------
# print("\nNo features: ")
# dataset_path = "nn/indico_splits/data_s1_rm1_lf0"
# criterions = ["book", "book-page", "subj"]
# for criterion in criterions:
#     print("\nCriterion: ", criterion)
#     print("X_train: ", np.load(os.path.join(dataset_path, criterion, 'X_train_' + criterion + '_0.npy')).shape)
#     print("y_train: ", np.load(os.path.join(dataset_path, criterion, 'y_train_' + criterion + '_0.npy')).shape)
#     print("X_test: ", np.load(os.path.join(dataset_path, criterion, 'X_test_' + criterion + '_0.npy')).shape)
#     print("y_test: ", np.load(os.path.join(dataset_path, criterion, 'y_test_' + criterion + '_0.npy')).shape)
#
