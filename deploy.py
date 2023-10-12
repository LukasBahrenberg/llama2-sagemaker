import os
from dotenv import load_dotenv

from transformers import AutoModel, AutoTokenizer

import sagemaker
from sagemaker import get_execution_role
from sagemaker.model import Model
from sagemaker.utils import name_from_base

load_dotenv()

AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')

model_name = "your-huggingface-model-name"  # replace with your model name
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

sagemaker_session = sagemaker.Session()
role = get_execution_role()

bucket = sagemaker_session.default_bucket()
prefix = 'your-prefix'  # e.g., 'huggingface-model'

model_artifact = sagemaker_session.upload_data(path='./model', bucket=bucket, key_prefix=prefix)

pytorch_model = Model(
    model_data=model_artifact,
    role=role,
    framework_version='1.9.0',  # use the appropriate PyTorch version
    py_version='py3',
    entry_point='inference.py',
    source_dir='code',
    sagemaker_session=sagemaker_session
)

predictor = pytorch_model.deploy(instance_type='ml.m5.large', initial_instance_count=1)