# from utils.dataloader import DataFetcher

# datafetcher = DataFetcher("data/train_list.txt", 'C:\\ORamaVR\\Datasets\\p2mppdata\\p2mppdata\\train', 'C:\\ORamaVR\\Datasets\\ShapeNetRendering')
from azureml.core import Dataset
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azureml.fsspec import AzureMachineLearningFileSystem

ml_client = MLClient.from_config(credential=DefaultAzureCredential())
data_asset = ml_client.data.get("test_working_data", version="1")

fs = AzureMachineLearningFileSystem(data_asset.path)
dat_file = fs.ls()[0]
print(dat_file)