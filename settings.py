import os
import getpass

username = getpass.getuser()

experiment_run = "001"

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

num_epochs = 50
batch_size = 3
img_size = 128
learning_rate = 1e-3

skip_empty = True
input_channels = 1
output_channels = 1