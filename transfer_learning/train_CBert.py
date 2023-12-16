from __future__ import absolute_import
import os
import torch
import string
import random
import logging
import numpy as np
from io import open
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler,TensorDataset
from transformers import ( AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          AutoConfig, AutoModel, AutoTokenizer)

from GCG_model import GitCommitGeneratorModel

seed = 42
n_gpu = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(seed)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                'bert': (AutoConfig, AutoModel, AutoTokenizer)}
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)



# convert the args into static variables 
MODEL_TYPE = "roberta"
MODEL_NAME = "microsoft/codebert-base"
OUTPUT_DIR = "output"
BATCH_SIZE = 128 
LEARNING_RATE = 5e-5
BEAM_SIZE = 10
WEIGHT_DECAY = 0.0
NUM_EPOCHS = 3.0
MAX_SOURCE_LEN = 512
MAX_TARGET_LEN = 50
ADAM_EPSILON = 1e-8
EVAL_STEP = 200
DO_TRAIN = True
DO_EVAL = True

train_batch_size = 32
eval_batch_size = 2
BASE_DATASET_PATH = 'filtered_dataset'
BASE_SPLIT_NAME = 'sort_random_train80_valid10_test10'
LANGUAGES = ['cpp', 'java', 'python', 'javascript', 'csharp']
DIFF_FILE_SUFFIX = '.diff.txt'
COMMIT_FILE_SUFFIX = '.msg.txt' 
# make dir if output_dir not exist
if os.path.exists(OUTPUT_DIR) is False:
    os.makedirs(OUTPUT_DIR)
        



class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target


def read_examples(diff_list,commit_list):
    """Read examples from filename."""
    examples=[]
    for idx,(diff,commit) in enumerate(zip(diff_list,commit_list)):
        examples.append(
            Example(
                idx = idx,
                source=diff,
                target = commit,
            )
        )
    return examples

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self,
                 idx,
                 source_ids,
                 target_ids,
                 source_mask,
                    target_mask,
                    ):  
        self.idx = idx
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def convert_examples_to_features(examples, tokenizer, max_source_length,max_target_length, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        source_tokens = tokenizer.tokenize(example.source)[:MAX_SOURCE_LEN-2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * len(source_ids)
        padding_length = max_source_length - len(source_ids)
        source_ids += ([tokenizer.pad_token_id] * padding_length)
        source_mask += ([0] * padding_length)
        target_tokens = tokenizer.tokenize(example.target)[:MAX_TARGET_LEN-2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        
        if stage == "test" : tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:MAX_TARGET_LEN-2]

        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = max_target_length - len(target_ids)
        target_ids += ([tokenizer.pad_token_id] * padding_length)
        target_mask += ([0] * padding_length)


        if example_index < 5 and stage == "train":
            logger.info("*** Example ***")
            logger.info("idx: {}".format(example.idx))

            logger.info("source_tokens: {}".format([x.replace('\u0120','_') for x in source_tokens]))
            logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
            logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))
            
            logger.info("target_tokens: {}".format([x.replace('\u0120','_') for x in target_tokens]))
            logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
            logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))


        features.append(
            InputFeatures(
                idx = example.idx,
                source_ids=source_ids,
                target_ids=target_ids,
                source_mask=source_mask,
                target_mask=target_mask,
            )
        )

    return features



def preprocess_word(word):
    word = word.lower()
    word = word.translate(str.maketrans('', '', string.punctuation))
    word = word.translate(str.maketrans('', '', string.digits))
    return word

def read_data(language, split_name):
    with open(os.path.join(BASE_DATASET_PATH, language, BASE_SPLIT_NAME, split_name+DIFF_FILE_SUFFIX), 'r') as diff_file, open(os.path.join(BASE_DATASET_PATH, language, BASE_SPLIT_NAME, split_name+COMMIT_FILE_SUFFIX), 'r') as commit_file:
        diff_lines = diff_file.readlines()
        diff_lines = [diff.strip() for diff in diff_lines]
        commit_lines = commit_file.readlines()
        commit_words = [line.strip().split() for line in commit_lines]
        commit_words = [word for line in commit_words for word in line]
        commit_words = [' '.join(preprocess_word(word) for word in commit_words)]
        return diff_lines, commit_lines
    
train_diffs, train_commit_messages = read_data('python', 'train')
valid_diffs, valid_commit_messages = read_data('python', 'valid')

print("train_diffs: ",len(train_diffs))
print("valid_diffs: ",len(valid_diffs))

config_class, model_class, tokenizer_class = MODEL_CLASSES[MODEL_TYPE]
config = config_class.from_pretrained(MODEL_NAME)
tokenizer = tokenizer_class.from_pretrained(MODEL_NAME)

#build model
encoder = model_class.from_pretrained(MODEL_NAME,config=config)    
decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)


model=GitCommitGeneratorModel(encoder=encoder,decoder=decoder,tokenizer=tokenizer, config=config,
            beam_size=BEAM_SIZE,max_length=32,
            sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)

print("model:\n",model)
model.to(device)

# ---------------------- Training & Evaluation ---------------------- #

train_examples = read_examples(train_diffs,train_commit_messages)
valid_examples = read_examples(valid_diffs,valid_commit_messages)

train_features = convert_examples_to_features(train_examples, tokenizer, MAX_SOURCE_LEN, MAX_TARGET_LEN, stage="train")
valid_features = convert_examples_to_features(valid_examples, tokenizer, MAX_SOURCE_LEN, MAX_TARGET_LEN, stage="valid")

train_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
train_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
train_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
train_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)
train_dataset = TensorDataset(train_source_ids, train_source_mask, train_target_ids, train_target_mask)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

 # Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay': WEIGHT_DECAY},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
t_total = len(train_dataloader) // NUM_EPOCHS
optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=ADAM_EPSILON)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

nb_tr_examples, nb_tr_steps,train_loss,global_step,best_bleu,best_loss = 0, 0,0,0,0,1e6 
model.train()

for epoch in range(int(NUM_EPOCHS)):
    bar = tqdm(train_dataloader, desc="Iteration")
    for batch in bar:
        batch = tuple(t.to(device) for t in batch)
        source_ids, source_mask, target_ids, target_mask = batch
        loss, _, _ = model(source_ids, target_ids, source_mask, target_mask)
        train_loss += loss.item()
        nb_tr_examples += source_ids.size(0)
        nb_tr_steps += 1
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        bar.set_description("epoch {} loss {}".format(epoch,train_loss))

        eval_loss = 0
        nb_eval_steps = 0

        if nb_tr_steps % EVAL_STEP == 0:
            model.eval()
            eval_examples = read_examples(valid_diffs,valid_commit_messages)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, MAX_SOURCE_LEN, MAX_TARGET_LEN, stage="valid")
            eval_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            eval_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
            eval_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
            eval_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)

            eval_dataset = TensorDataset(eval_source_ids, eval_source_mask, eval_target_ids, eval_target_mask)
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

            for batch in eval_dataloader:
                batch = tuple(t.to(device) for t in batch)
                source_ids, source_mask, target_ids, target_mask = batch
                with torch.no_grad():
                    loss, _, _ = model(source_ids, target_ids, source_mask, target_mask)
                    eval_loss += loss.mean().item()
                nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            logger.info("eval_loss: {}".format(eval_loss))

            if eval_loss < best_loss:
                best_loss = eval_loss
                logger.info("Saving model with best loss: {}".format(best_loss))
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_loss_model.bin"))

            model.train()

    # save checkpoint after each epoch
    logger.info("Saving model checkpoint to %s", OUTPUT_DIR)
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), os.path.join(OUTPUT_DIR, "pytorch_model.bin".format(epoch)))
    tokenizer.save_pretrained(OUTPUT_DIR)
    torch.save(optimizer.state_dict(), os.path.join(OUTPUT_DIR, "optimizer.pt".format(epoch)))
    torch.save(scheduler.state_dict(), os.path.join(OUTPUT_DIR, "scheduler.pt".format(epoch)))
    logger.info("Saving model checkpoint to %s", OUTPUT_DIR)

    