from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model
import torch


# 将JSON文件转换为CSV文件
df = pd.read_json('fake_sft.json')
ds = Dataset.from_pandas(df)
ds[:3]