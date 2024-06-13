import os
import random
from os.path import join
from tqdm import tqdm
from collections import defaultdict as dd
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tqdm import trange
from sklearn.metrics import classification_report, precision_recall_fscore_support, average_precision_score
import logging
from cogdl.oag import oagbert


import json
import pandas as pd

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


MAX_SEQ_LENGTH=512
DATA_TRACE_DIR = 'data'
OUT_DIR = 'out'

def load_json(rfdir, rfname):
    with open(join(rfdir, rfname), 'r', encoding='utf-8') as rf:
        data = json.load(rf)
        return data
    
    
def dump_json(obj, wfdir, wfname):
    with open(join(wfdir, wfname), 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, indent=4, ensure_ascii=False)
        
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def prepare_bert_input():

    data_dir = join(DATA_TRACE_DIR, "PST")
    papers = load_json(data_dir, "paper_source_trace_train_ans.json")
    n_papers = len(papers)
    papers = sorted(papers, key=lambda x: x["_id"])
    n_valid = int(n_papers * 1 / 4)
    sc = load_json(data_dir, "smartcite2.json")
    for i in range(4):
        all_y = []
        x_train = []
        y_train = []
        x_valid = []
        y_valid = []
        fold=i+1
        # n_valid = n_papers - n_train
        if fold==1:
            papers_train = papers[n_valid:]
            papers_valid = papers[:n_valid]
        elif fold==2:
            papers_train = papers[:n_valid]+papers[2*n_valid:]
            papers_valid = papers[n_valid:2*n_valid]
        elif fold==3:
            papers_train = papers[:2*n_valid]+papers[3*n_valid:]
            papers_valid = papers[2*n_valid:3*n_valid]
        elif fold==4:
            papers_train = papers[:3*n_valid]
            papers_valid = papers[3*n_valid:]
        print(f'----------fold {i} trainSet {len(papers_train)} testSet {len(papers_valid)}')
        pids_train = sorted({p["_id"] for p in papers_train})
        pids_valid = sorted({p["_id"] for p in papers_valid})
        dfx = pd.read_csv('out/kddcup/rf/train_data.csv')
        pos_set = set()
        neg_set = set()
        for i in range(len(dfx)):
            if dfx.loc[i,'label']==1 and dfx.loc[i,'mpos'] != 2:
                pos_set.add(dfx.loc[i,'id']+str(dfx.loc[i,'ref_idx']))
            if dfx.loc[i,'label']==0 and dfx.loc[i,'mpos'] != 2:
                neg_set.add(dfx.loc[i,'id']+str(dfx.loc[i,'ref_idx']))
        for paper in papers:
            pid = paper['_id']
            ds = x_train
            dl = y_train
            if paper in papers_valid:
                ds = x_valid
                dl = y_valid
            for refkey in sc[pid]:
                data = sc[pid][refkey]
                refidx = int(refkey[1:])
                if pid+str(refidx) in pos_set or pid+str(refidx) in neg_set :
                    texts = []
                    for i,v in enumerate(data['refNum']):
                        if v == 1:
                            texts.append(data['texts'][i])
                    if len(texts)>0:
                        ds.append(' '.join(texts))
                        dl.append(1 if pid+str(refidx) in pos_set else 0)
        print("len(x_train)", len(x_train), "len(x_valid)", len(x_valid))
        print(len(x_train),len(y_train),sum(y_train),len(x_valid),len(y_valid),sum(y_valid))
        with open(join(data_dir, f"sig2_bib_context_train_{fold}.txt"), "w", encoding="utf-8") as f:
            for line in x_train:
                f.write(line + "\n")
        with open(join(data_dir, f"sig2_bib_context_valid_{fold}.txt"), "w", encoding="utf-8") as f:
            for line in x_valid:
                f.write(line + "\n")

        with open(join(data_dir, f"sig2_bib_context_train_label_{fold}.txt"), "w", encoding="utf-8") as f:
            for line in y_train:
                f.write(str(line) + "\n")

        with open(join(data_dir, f"sig2_bib_context_valid_label_{fold}.txt"), "w", encoding="utf-8") as f:
            for line in y_valid:
                f.write(str(line) + "\n")


class BertInputItem(object):
    """An item with all the necessary attributes for finetuning BERT."""

    def __init__(self, text, input_ids, input_mask, segment_ids, label_id):
        self.text = text
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_examples_to_inputs(example_texts, example_labels, max_seq_length, tokenizer, verbose=0):
    """Loads a data file into a list of `InputBatch`s."""
    
    input_items = []
    examples = zip(example_texts, example_labels)
    for (ex_index, (text, label)) in enumerate(examples):

        # Create a list of token ids
        input_ids = tokenizer.encode(f"[CLS] {text} [SEP]")
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]

        # All our tokens are in the first input segment (id 0).
        segment_ids = [0] * len(input_ids)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label

        input_items.append(
            BertInputItem(text=text,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
        
    return input_items


def get_data_loader(features, max_seq_length, batch_size, shuffle=True): 

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)
    return dataloader


def evaluate(model, dataloader, device, criterion):
    model.eval()
    
    eval_loss = 0
    nb_eval_steps = 0
    predicted_labels, correct_labels = [], []

    for step, batch in enumerate(tqdm(dataloader, desc="Evaluation iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            r = model(input_ids, attention_mask=input_mask,
                                          token_type_ids=segment_ids, labels=label_ids)
            # tmp_eval_loss = r[0]
            logits = r[1]
            # print("logits", logits)
            tmp_eval_loss = criterion(logits, label_ids)

        outputs = np.argmax(logits.to('cpu'), axis=1)
        label_ids = label_ids.to('cpu').numpy()
        
        predicted_labels += list(outputs)
        correct_labels += list(label_ids)
        
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    
    correct_labels = np.array(correct_labels)
    predicted_labels = np.array(predicted_labels)
        
    return eval_loss, correct_labels, predicted_labels


def train(fold,year=2023, model_name="scibert"):
    print(f"----------startTrain fold{fold},model name {model_name}")
    train_texts = []
    dev_texts = []
    train_labels = []
    dev_labels = []
    data_year_dir = join(DATA_TRACE_DIR, "PST")
    print("data_year_dir", data_year_dir)

    with open(join(data_year_dir, f"sig2_bib_context_train_{fold}.txt"), "r", encoding="utf-8") as f:
        for line in f:
            train_texts.append(line.strip())
    with open(join(data_year_dir, f"sig2_bib_context_valid_{fold}.txt"), "r", encoding="utf-8") as f:
        for line in f:
            dev_texts.append(line.strip())

    with open(join(data_year_dir, f"sig2_bib_context_train_label_{fold}.txt"), "r", encoding="utf-8") as f:
        for line in f:
            train_labels.append(int(line.strip()))
    with open(join(data_year_dir, f"sig2_bib_context_valid_label_{fold}.txt"), "r", encoding="utf-8") as f:
        for line in f:
            dev_labels.append(int(line.strip()))

    print("Train size:", len(train_texts))
    print("Dev size:", len(dev_texts))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_weight = len(train_labels) / (2 * np.bincount(train_labels))
    class_weight = torch.Tensor(class_weight).to(device)
    print("Class weight:", class_weight)

    if model_name == "bert":
        BERT_MODEL = "bert-base-uncased"
    elif model_name == "scibert":
        BERT_MODEL = "allenai/scibert_scivocab_uncased"
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
        model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels = 2)
    elif model_name == "oagbert":
        tokenizer, model2 = oagbert('oagbert-v2')
        model = BertForSequenceClassification.from_pretrained('/home/featurize/work/kddpst/paper-source-trace/saved/oagbert-v2', num_labels = 2)
    else:
        raise NotImplementedError


    model.to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weight)

    train_features = convert_examples_to_inputs(train_texts, train_labels, MAX_SEQ_LENGTH, tokenizer, verbose=0)
    dev_features = convert_examples_to_inputs(dev_texts, dev_labels, MAX_SEQ_LENGTH, tokenizer)

    BATCH_SIZE = 32
    train_dataloader = get_data_loader(train_features, MAX_SEQ_LENGTH, BATCH_SIZE, shuffle=True)
    dev_dataloader = get_data_loader(dev_features, MAX_SEQ_LENGTH, BATCH_SIZE, shuffle=False)

    GRADIENT_ACCUMULATION_STEPS = 1
    NUM_TRAIN_EPOCHS = 20
    LEARNING_RATE = 1e-6
    WARMUP_PROPORTION = 0.1
    MAX_GRAD_NORM = 5

    num_train_steps = int(len(train_dataloader.dataset) / BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(WARMUP_PROPORTION * num_train_steps)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)

    OUTPUT_DIR = join(OUT_DIR, "kddcup", model_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    MODEL_FILE_NAME = f"pytorch_model_sig_union_{fold}.bin"
    PATIENCE = 5

    loss_history = []
    no_improvement = 0
    for _ in trange(int(NUM_TRAIN_EPOCHS), desc="Epoch"):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            outputs = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)
            # loss = outputs[0]
            logits = outputs[1]

            loss = criterion(logits, label_ids)

            if GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)  
                
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
        dev_loss, _, _ = evaluate(model, dev_dataloader, device, criterion)
        
        print("Loss history:", loss_history)
        print("Dev loss:", dev_loss)
        
        if len(loss_history) == 0 or dev_loss < min(loss_history):
            no_improvement = 0
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(OUTPUT_DIR, MODEL_FILE_NAME)
            torch.save(model_to_save.state_dict(), output_model_file)
        else:
            no_improvement += 1
        
        if no_improvement >= PATIENCE: 
            print("No improvement on development set. Finish training.")
            break
            
        loss_history.append(dev_loss)
    
    

def gen_kddcup_submission_bert(fold,fkey="train",sfile= "paper_source_trace_train_ans.json"):
    model_name = 'scibert'
    print("model name", model_name)
    data_dir = join(DATA_TRACE_DIR, "PST")
    papers = load_json(data_dir,sfile)
    BERT_MODEL = "allenai/scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels = 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    model.load_state_dict(torch.load(join(OUT_DIR, "kddcup", model_name, f"pytorch_model_sig_union_{fold}.bin")))
    model.to(device)
    model.eval()
    BATCH_SIZE = 32
    xml_dir = join(data_dir, "paper-xml")
    sub_dict = {}
    sc = load_json(data_dir, "smartcite2.json")
    dfx = pd.read_csv('out/kddcup/rf/train_data.csv')
    valid_set = set()
    for i in range(len(dfx)):
        if dfx.loc[i,'mpos'] != 2:
            valid_set.add(dfx.loc[i,'id']+str(dfx.loc[i,'ref_idx']))
    scoreMap = {}
    for paper in tqdm(papers):
        cur_pid = paper["_id"]
        pid = cur_pid
        ds = []
        idxs = []
        scoreMap[cur_pid] = {}
        for refkey in sc[cur_pid]:
            data = sc[cur_pid][refkey]
            refidx = int(refkey[1:])
            if pid+str(refidx) in valid_set:
                scoreMap[cur_pid][str(refidx)] = 0.0
                texts = []
                for i,v in enumerate(data['refNum']):
                    if v == 1:
                        texts.append(data['texts'][i])
                if len(texts)>0:
                    ds.append(' '.join(texts))
                    idxs.append(refidx)
        y_score = [0] * len(ds)
        test_features = convert_examples_to_inputs(ds, y_score, MAX_SEQ_LENGTH, tokenizer)
        test_dataloader = get_data_loader(test_features, MAX_SEQ_LENGTH, BATCH_SIZE, shuffle=False)
        predicted_scores = []
        for step, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                r = model(input_ids, attention_mask=input_mask,
                                            token_type_ids=segment_ids, labels=label_ids)
                tmp_eval_loss = r[0]
                logits = r[1]
            cur_pred_scores = logits[:, 1].to('cpu').numpy()
            predicted_scores.extend(cur_pred_scores)
        for ii in range(len(predicted_scores)):
            scoreMap[cur_pid][str(idxs[ii])] = float(sigmoid(predicted_scores[ii]))
    dump_json(scoreMap, join(OUT_DIR, "kddcup", model_name), f"{fkey}_sig_union_bert_{fold}.json")

    
def set_random_seed(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    set_random_seed(1995)
    prepare_bert_input()
    for i in range(4):
        train(model_name="scibert",fold=i+1)
        gen_kddcup_submission_bert(fold=i+1,fkey='train',sfile= "paper_source_trace_train_ans.json")
        gen_kddcup_submission_bert(fold=i+1,fkey='valid',sfile= "paper_source_trace_valid_wo_ans.json")
        gen_kddcup_submission_bert(fold=i+1,fkey='test',sfile= "paper_source_trace_test_wo_ans.json")
