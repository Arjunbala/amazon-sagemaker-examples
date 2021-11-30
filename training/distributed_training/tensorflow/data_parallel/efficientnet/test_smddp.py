import sagemaker
from sagemaker import get_execution_role
from sagemaker.estimator import Estimator
import boto3
import os
from sagemaker.tensorflow import TensorFlow
from sagemaker.inputs import FileSystemInput

sagemaker_session = sagemaker.Session()
bucket = sagemaker_session.default_bucket()

role = (
    get_execution_role()
)  # provide a pre-existing role ARN as an alternative to creating a new role
role_name = role.split(["/"][-1])
print(f"SageMaker Execution Role:{role}")
print(f"The name of the Execution role: {role_name[-1]}")

client = boto3.client("sts")
account = client.get_caller_identity()["Account"]
print(f"AWS account:{account}")

session = boto3.session.Session()
region = session.region_name
print(f"AWS region:{region}")

image = "efficientnet-smdataparallel-sagemaker"  # Example: mask-rcnn-smdataparallel-sagemaker
tag = "tf2.6-with-tfio"  # Example: pt1.8

instance_type = "ml.p4d.24xlarge"
instance_count = 2
docker_image = f"{account}.dkr.ecr.{region}.amazonaws.com/{image}:{tag}"
username = "AWS"
job_name = "tf-smdataparallel-efficientnet-try5"
print(docker_image)


# FSX settings
subnets = ["subnet-0a74b4dd29646e1a0"]
security_group_ids = [
    "sg-03cb5da33ff05fdd0"
]
file_system_id = "fs-0619a595d6d561152"
file_system_directory_path = (
    "/5uhyvbmv/fsx/efficientnet/tf"  # NOTE: '/fsx/' will be the root mount path. Example: '/fsx/mask_rcnn/PyTorch'
)
file_system_access_mode = "ro"
file_system_type = "FSxLustre"
train_fs = FileSystemInput(
    file_system_id=file_system_id,
    file_system_type=file_system_type,
    directory_path=file_system_directory_path,
    file_system_access_mode=file_system_access_mode,
)
data_channels = {"train": train_fs}

# Configure the hyper-parameters
hyperparameters = {
    "max_epochs" : 10,
    "mode" : "train",
    "arch": "efficientnet-b0",
    "save_checkpoint_freq": 100
}

# Configure metrics
metric_definitions = [
    {"Name": "train_throughput", "Regex": "examples/second : (.*?) "},
]

"""
estimator = TensorFlow(
    entry_point="main.py",
    role=role,
    image_uri=docker_image,
    source_dir="./smddp-adapt/efficientnet",
    instance_count=instance_count,
    instance_type=instance_type,
    framework_version="2.6",
    py_version="py38",
    sagemaker_session=sagemaker_session,
    hyperparameters=hyperparameters,
    subnets=subnets,
    security_group_ids=security_group_ids,
    debugger_hook_config=False,
    # Training using SMDataParallel Distributed Training Framework
    distribution={"smdistributed": {"dataparallel": {"enabled": True}}},
)
"""

estimator = TensorFlow(
    entry_point="main.py",
    role=role,
    image_uri=docker_image,
    source_dir="./smddp-adapt/efficientnet",
    instance_count=instance_count,
    instance_type=instance_type,
    framework_version="2.6",
    py_version="py38",
    sagemaker_session=sagemaker_session,
    hyperparameters=hyperparameters,
    #subnets=subnets,
    #security_group_ids=security_group_ids,
    debugger_hook_config=False,
    metric_definitions = metric_definitions,
    # Training using SMDataParallel Distributed Training Framework
    distribution={"smdistributed": {"dataparallel": {"enabled": True}}},
)
estimator.fit(inputs={"train":"s3://smddp-570106654206-us-west-2/dataset/efficient/"},job_name=job_name)
#estimator.fit(inputs=data_channels,job_name=job_name)
