import os
import getpass

username = getpass.getuser()

experiment_run = "021_cross_validation_dropout"

if "COLAB_GPU" in os.environ:
    data_path = "/content/ml4h_proj1_colon_cancer_ct/"
    out_dir = "/content/drive/MyDrive/ML4H/"
    colab = True
else:
    data_path = "/cluster/scratch/{}/ML4H/ml4h_proj1_colon_cancer_ct/".format(username)
    out_dir = "/cluster/scratch/{}/ML4H/saved_models/".format(username)
    colab = False

model_dir = os.path.join(out_dir, experiment_run)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_file = os.path.join(model_dir, "bestmodel.pt")
chkpoint_file = os.path.join(model_dir, "chkpoint_")

train_dir = os.path.join(data_path, "imagesTr")
labels_dir = os.path.join(data_path, "labelsTr")
test_dir = os.path.join(data_path, "imagesTs")
predictions_dir = os.path.join(data_path, "predictionsTs")
if not os.path.exists(predictions_dir):
    os.mkdir(predictions_dir)

num_epochs = 24
batch_size = 16
test_batch_size = 1
img_size = 512
learning_rate = 1e-3

downsample = True
upsample = False
shuffle_files = True
input_channels = 1
output_channels = 1

train_val_splitting_ratio = 0.8
seed = 1234

with_dropout = True
drop_prob = 0.2

max_epochs_no_improve = 12
k_folds = 3

# Ensemble:
ensemble = True

# Without Dropout
"""
pred_dir = os.path.join(out_dir, "020_cross_validation_50")
ensemble_model_1 = os.path.join(pred_dir, "bestmodel_fold1.pt")
ensemble_model_2 = os.path.join(pred_dir, "bestmodel_fold2.pt")
ensemble_model_3 = os.path.join(pred_dir, "bestmodel_fold3.pt")
"""

# With Dropout
pred_dir_dp = os.path.join(out_dir, "021_cross_validation_dropout")
ensemble_model_1 = os.path.join(pred_dir_dp, "chkpoint_bestmodel_fold1.pt")
ensemble_model_2 = os.path.join(pred_dir_dp, "chkpoint_bestmodel_fold2.pt")
ensemble_model_3 = os.path.join(pred_dir_dp, "chkpoint_bestmodel_fold3.pt")
