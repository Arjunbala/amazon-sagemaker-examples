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
job_name = "tf-horovod-efficientnet-try2"
print(docker_image)

# Configure the hyper-parameters
hyperparameters = {
    "max_epochs" : 10,
    "mode" : "train",
    "arch": "efficientnet-b0",
    "save_checkpoint_freq": 100
}

# Horovod specific
hvd_processes_per_host = 8

distributions = {'mpi': {
                    'enabled': True,
                    'processes_per_host': hvd_processes_per_host,
                    'custom_mpi_options': '-verbose --NCCL_DEBUG=INFO -x OMPI_MCA_btl_vader_single_copy_mechanism=none'
                        }
                }

estimator = TensorFlow(
    entry_point="main.py",
    role=role,
    image_uri=docker_image,
    source_dir="./horovod-adapt/efficientnet",
    instance_count=instance_count,
    instance_type=instance_type,
    framework_version="2.6",
    py_version="py38",
    sagemaker_session=sagemaker_session,
    hyperparameters=hyperparameters,
    debugger_hook_config=False,
    # Training using Horovod
    distribution=distributions,
)
estimator.fit(inputs={"train":"s3://smddp-570106654206-us-west-2/dataset/efficient/"},job_name=job_name)
#estimator.fit(inputs=data_channels,job_name=job_name)
