#%%
import nncf  # noqa: F811
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import openvino as ov


#%%
calibDataset = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Datas", "Calibration")
modelDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Models", "YoloPoseV8")
modelName = "yolov8n-pose.onnx"


#%%
class CalibrationDataset(Dataset):
    def __init__(self, folder:str) -> None:
        self.folder = folder
        self.filelist = [ x for x in os.listdir(self.folder) if x.endswith('.jpg') or x.endswith('png') ]
        print("Found ", len(self.filelist), " images in Calibration dataset: ", folder)

    def __getitem__(self, index):
        img = np.array(Image.open(os.path.join(self.folder, self.filelist[index])).convert('RGB').resize((640, 640)))
        return img
    
    def __len__(self):
        return len(self.filelist)

print(calibDataset)
print(modelDir)

#%%
def transform_fn(data_item):
    """
    Extract the model's input from the data item.
    The data item here is the data item that is returned from the data source per iteration.
    This function should be passed when the data item cannot be used as model's input.
    """
    img = np.array(data_item, dtype=np.int64).transpose(0, 3, 2, 1)
    return img

dataset = CalibrationDataset(calibDataset)
dataloader = DataLoader(dataset, 1, False)
quantization_dataset = nncf.Dataset(dataloader, transform_fn)

# %%
# ignored_scope = nncf.IgnoredScope(
#     types=["Multiply", "Subtract", "Sigmoid"],  # ignore operations
#     names=[
#         "/model.22/dfl/conv/Conv",           # in the post-processing subgraph
#         "/model.22/Add",
#         "/model.22/Add_1",
#         "/model.22/Add_2",
#         "/model.22/Add_3",
#         "/model.22/Add_4",   
#         "/model.22/Add_5",
#         "/model.22/Add_6",
#         "/model.22/Add_7",
#         "/model.22/Add_8",
#         "/model.22/Add_9",
#         "/model.22/Add_10"
#     ]
# )


# Detection model
core = ov.Core()
pose_ov_model = core.read_model(os.path.join(modelDir, "FP16", "yolov8n-pose.xml"))
quantized_pose_model = nncf.quantize(
    pose_ov_model,
    quantization_dataset,
    preset=nncf.QuantizationPreset.MIXED,
    #ignored_scope=ignored_scope
)
# %%
from openvino.runtime import serialize
int8_model_pose_path = "OpenVINOTest/Models/YoloPoseV8/Int8/yolov8n-pose.xml"
print(f"Quantized keypoint detection model will be saved to {int8_model_pose_path}")
serialize(quantized_pose_model, str(int8_model_pose_path))
# %%


#%%
import nncf  # noqa: F811
from torchvision import datasets, transforms
import torch
import openvino.runtime as ov

model = ov.Core().read_model(os.path.join(modelDir, modelName))

# Provide validation part of the dataset to collect statistics needed for the compression algorithm
val_dataset = datasets.ImageFolder(calibDataset, transform=transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()]))
dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

# Step 1: Initialize transformation function
def transform_fn(data_item):
    images, _ = data_item
    return images

# Step 2: Initialize NNCF Dataset
calibration_dataset = nncf.Dataset(dataset_loader, transform_fn)
# Step 3: Run the quantization pipeline
quantized_model = nncf.quantize(model, calibration_dataset)
# %%
from openvino.runtime import serialize
int8_model_pose_path = "OpenVINOTest/Models/YoloPoseV8/Int8_2/yolov8n-pose.xml"
print(f"Quantized keypoint detection model will be saved to {int8_model_pose_path}")
serialize(quantized_model, str(int8_model_pose_path))
# %%
