import os
from dotenv import load_dotenv

from transformers import AutoModel, AutoTokenizer

import sagemaker
from sagemaker import get_execution_role
from sagemaker.model import Model
from sagemaker.utils import name_from_base
from sagemaker.image_uris import retrieve

load_dotenv()

HUGGING_FACE_ACCESS_KEY = os.environ.get('HUGGING_FACE_ACCESS_KEY')

model_name = "meta-llama/Llama-2-7b-chat-hf"  # replace with your model name

model = AutoModel.from_pretrained(model_name, token=HUGGING_FACE_ACCESS_KEY)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGING_FACE_ACCESS_KEY)

model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

sagemaker_session = sagemaker.Session()
role = get_execution_role()

bucket = sagemaker_session.default_bucket()
prefix = 'your-prefix'  # e.g., 'huggingface-model'

model_artifact = sagemaker_session.upload_data(path='./model', bucket=bucket, key_prefix=prefix)

region = sagemaker.Session().boto_region_name
pytorch_container = retrieve('pytorch', region, version='1.9.0')  # Replace '1.9.0' with your desired version

pytorch_model = Model(
    model_data=model_artifact,
    role=role,
    image_uri=pytorch_container,
    entry_point='inference.py',
    source_dir='code',
    sagemaker_session=sagemaker_session
)

predictor = pytorch_model.deploy(instance_type='ml.m5.large', initial_instance_count=1)