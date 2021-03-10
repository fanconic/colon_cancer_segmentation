import os
import getpass

username = getpass.getuser()

experiment_run = "001"

if "COLAB_GPU" in os.environ:
    data_path = "/content/ml4h_proj1_colon_cancer_ct/"
    out_dir = "/content/drive/MyDrive/ML4H/"
    colab = True
else:
    data_path = "/scratch/ML4H/ml4h_proj1_colon_cancer_ct/"
    out_dir = os.path.join("/cluster/scratch/", username, "/ML4H/saved_models/")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    colab = False

model_dir = os.path.join(out_dir, experiment_run)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_file = os.path.join(model_dir, "bestmodel.pt")
chkpoint_file = os.path.join(model_dir, "chkpoint_")

train_dir = os.path.join(data_path, "imagesTr")
labels_dir = os.path.join(data_path, "imagesTr")
test_dir = os.path.join(data_path, "imagesTs")

num_epochs = 25
batch_size = 3
img_size = 256
learning_rate = 1e-3

skip_empty = True
input_channels = 1
output_channels = 1