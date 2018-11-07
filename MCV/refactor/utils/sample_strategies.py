import pandas as pd
import numpy as np
import io
import gc
import boto3
import re

# custom functions
from utils import *


# s3 work path
bucket = 'boi-banregio'
prefix = 'datalake/data/InteligenciaRiesgos/M&M/MCV'
# s3 resource
s3_bucket_resource = boto3.resource('s3').Bucket(bucket)

# rfc & fecha & llave dataset
rfc_key_features_file_name = "DATASETS_3/JAT_MCV_VAR_VARIABLES_SELECCION/PARTITION/EXTRA_FEATURES/MCV_VAR_RFC_FECHA_LLAVE.csv"
rfc_key_features_file_key = "{}/{}".format(prefix, rfc_key_features_file_name)
rfc_key_features_obj = s3_bucket_resource.Object(rfc_key_features_file_key).get()
rfc_key_features_data = pd.read_csv(io.BytesIO(rfc_key_features_obj['Body'].read()))

# strategy 1
periods = [
    [201501, 201502, 201503, 201504, 201505],
    [201508, 201509, 201510, 201511, 201512, 201601, 201602, 201603, 201604, 201605],
    [201608, 201609, 201610, 201611, 201612, 201701, 201702, 201703, 201704, 201705]  
]

sample = pd.Series([False]*rfc_key_features_data.shape[0])
period_samples = []
sample_excludes = [sample]*2

stg1_sample, stg1_period_samples, stg1_sample_exclude = sampler1(
    rfc_key_features_data, 
    periods, 
    sample, 
    period_samples, 
    sample_excludes, 
    month_size=[
        2219, 2500, 2000, 2000, 2500, 
        2448, 2000, 3000, 2500, 2100, 3000, 2800, 2200, 3000, 3000, 
        1464, 3500, 4000, 1300, 3000, 3000, 3222, 3000, 3000, 3000
    ]
)


# startegy 2
periods = [
    [201501],
    [201504, 201505],
    [201508, 201509],
    [201512, 201601],
    [201604, 201605],
    [201608, 201609],
    [201612, 201701], 
    [201704, 201705]
]

sample = pd.Series([False]*rfc_key_features_data.shape[0])
period_samples = []
sample_exclude = sample

stg2_sample, stg2_period_samples, stg2_sample_exclude = sampler(
    rfc_key_features_data, 
    periods, 
    sample, 
    period_samples, 
    sample_exclude, 
    month_size=[
        6460,
        2388, 4500,
        2793, 4500, 
        3261, 4500, 
        3695, 4500, 
        4261, 4500, 
        4561, 4500, 
        4205, 5000
    ]
)



