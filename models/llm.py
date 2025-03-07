import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch import nn

from torch_geometric.nn.pool import global_add_pool
from transformers import (LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoModel)
from peft import LoftQConfig, LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig
from accelerate.hooks import remove_hook_from_module
from utils import get_available_devices
from tqdm.autonotebook import trange
import gc

LLM_DIM_DICT = {"PubmedBERT": 768, "ST": 768, "BERT": 768, "e5": 1024, "llama2_7b": 4096, "llama2_13b": 5120}


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class LLMModel(torch.nn.Module):
    """
    Large language model from transformers.
    If peft is ture, use lora with pre-defined parameter setting for efficient fine-tuning.
    quantization is set to 4bit and should be used in the most of the case to avoid OOM.
    """
    def __init__(self, llm_name, quantization=True, peft=True, cache_dir="cache_data/model", max_length=500):
        super().__init__()
        assert llm_name in LLM_DIM_DICT.keys()
        self.llm_name = llm_name
        self.quantization = quantization

        self.indim = LLM_DIM_DICT[self.llm_name]
        self.cache_dir = cache_dir
        self.max_length = max_length
        model, self.tokenizer = self.get_llm_model()
        if peft:
            self.model = self.get_lora_perf(model)
        else:
            self.model = model
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = 'right'

    def find_all_linear_names(self, model):
        """
        find all module for LoRA fine-tuning.
        """
        cls = bnb.nn.Linear4bit if self.quantization else torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if 'lm_head' in lora_module_names:  # needed for 16-bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    def create_bnb_config(self):
        """
        quantization configuration.
        """
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        return bnb_config

    def get_lora_perf(self, model):
        """
        LoRA configuration.
        """
        target_modules = self.find_all_linear_names(model)
        config = LoraConfig(
            target_modules=target_modules,
            r=32,  # dimension of the updated matrices
            lora_alpha=16,  # parameter for scaling
            lora_dropout=0.2,  # dropout probability for layers
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        model = get_peft_model(model, config)
        model = prepare_model_for_kbit_training(model)

        return model

    def get_llm_model(self):
        if self.llm_name == "llama2_7b":
            model_name = "meta-llama/Llama-2-7b-hf"
            ModelClass = LlamaForCausalLM
            TokenizerClass = LlamaTokenizer

        elif self.llm_name == "llama2_13b":
            model_name = "meta-llama/Llama-2-13b-hf"
            ModelClass = LlamaForCausalLM
            TokenizerClass = LlamaTokenizer

        elif self.llm_name == "e5":
            model_name = "intfloat/e5-large-v2"
            ModelClass = AutoModel
            TokenizerClass = AutoTokenizer

        elif self.llm_name == "BERT":
            model_name = "bert-base-uncased"
            ModelClass = AutoModel
            TokenizerClass = AutoTokenizer

        elif self.llm_name == "ST":
            model_name = "sentence-transformers/multi-qa-distilbert-cos-v1"
            ModelClass = AutoModel
            TokenizerClass = AutoTokenizer

        elif self.llm_name == "PubmedBERT":
            model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
            ModelClass = AutoModel
            TokenizerClass = AutoTokenizer

        else:
            raise ValueError(f"Unknown language model: {self.llm_name}.")
        if self.quantization:
            bnb_config = self.create_bnb_config()
            model = ModelClass.from_pretrained(model_name,
                                               quantization_config=bnb_config,
                                               #attn_implementation="flash_attention_2",
                                               cache_dir=self.cache_dir)
        else:
            model = ModelClass.from_pretrained(model_name, cache_dir=self.cache_dir)
        model = remove_hook_from_module(model, recurse=True)
        tokenizer = TokenizerClass.from_pretrained(model_name, cache_dir=self.cache_dir, add_eos_token=True)
        if self.llm_name[:6] == "llama2":
            tokenizer.pad_token = tokenizer.bos_token
        return model, tokenizer

    def pooling(self, outputs, text_tokens=None):
        if self.llm_name in ["BERT", "ST", "PubmedBERT", "e5"]:
            return F.normalize(mean_pooling(outputs,
                                text_tokens.attention_mask), p=2, dim=1)
        else:
            return outputs[text_tokens.input_ids == 2] # llama2 EOS token

    def forward(self, text_tokens):

        outputs = self.model(input_ids=text_tokens["input_ids"],
                             attention_mask=text_tokens["attention_mask"],
                             output_hidden_states=True,
                             return_dict=True)["hidden_states"][-1]
        return self.pooling(outputs, text_tokens)

    def encode(self, text_tokens, pooling=False):

        with torch.no_grad():
            outputs = self.model(input_ids=text_tokens["input_ids"],
                                 attention_mask=text_tokens["attention_mask"],
                                 output_hidden_states=True,
                                 return_dict=True)["hidden_states"][-1]
            outputs = outputs.to(torch.float32)
            if pooling:
                outputs = self.pooling(outputs, text_tokens)

            return outputs, text_tokens.attention_mask


class SentenceEncoder:
    def __init__(self, llm_name, cache_dir="cache_data/model", batch_size=1, multi_gpu=False, max_lengths=1000):
        self.llm_name = llm_name
        self.device, _ = get_available_devices()
        self.batch_size = batch_size
        self.multi_gpu = multi_gpu
        self.max_lengths = max_lengths
        self.model = LLMModel(llm_name, quantization=False, peft=False, cache_dir=cache_dir, max_length=max_lengths)
        self.model.to(self.device)

    def encode(self, texts, to_tensor=True):
        all_embeddings = []
        with torch.no_grad():
            for start_index in trange(0, len(texts), self.batch_size, desc="Batches", disable=False, ):
                sentences_batch = texts[start_index: start_index + self.batch_size]
                text_tokens = self.model.tokenizer(sentences_batch, return_tensors="pt", padding="longest", truncation=True,
                                           max_length=self.max_lengths).to(self.device)
                embeddings, _ = self.model.encode(text_tokens, pooling=True)
                embeddings = embeddings.cpu()
                all_embeddings.append(embeddings)
        all_embeddings = torch.cat(all_embeddings, dim=0)
        if not to_tensor:
            all_embeddings = all_embeddings.numpy()

        return all_embeddings

    def flush_model(self):
        # delete llm from gpu to save GPU memory
        if self.model is not None:
            self.model = None
        gc.collect()
        torch.cuda.empty_cache()