# oagbert
import os
from cogdl.oag import oagbert
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import sys
from os.path import join
import json

def load_json(rfdir, rfname):
    with open(join(rfdir, rfname), 'r', encoding='utf-8') as rf:
        data = json.load(rf)
        return data
    
    
def dump_json(obj, wfdir, wfname):
    with open(join(wfdir, wfname), 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, indent=4, ensure_ascii=False)

pdmap = {}
ebdmap = {}
dblp_fname = "data/dblpmerge.json"
with open( dblp_fname, "r", encoding="utf-8") as myFile:
    for i, line in tqdm(enumerate(myFile), desc="readDBLP", total=30000):
        if len(line) <= 10:
            continue
        paper_tmp = json.loads(line.strip())
        pdmap[paper_tmp['id']] = paper_tmp


def get_ebd(model,bert,pid):
    # encode first paper
    if pid in ebdmap:
        return ebdmap[pid]
    pdict  = pdmap[pid]
    input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, position_ids_second, masked_positions, num_spans = model.build_inputs(
        title=pdict['title'], abstract=pdict['abstract'], venue=pdict['venue'], authors=[item['name'] for item in pdict['authors']], concepts=pdict['keywords'], affiliations=''
    )
    _, paper_embed_1 = model.bert.forward(
        input_ids=torch.LongTensor(input_ids).unsqueeze(0),
        token_type_ids=torch.LongTensor(token_type_ids).unsqueeze(0),
        attention_mask=torch.LongTensor(input_masks).unsqueeze(0),
        output_all_encoded_layers=False,
        checkpoint_activations=False,
        position_ids=torch.LongTensor(position_ids).unsqueeze(0),
        position_ids_second=torch.LongTensor(position_ids_second).unsqueeze(0)
    )
    ebd = F.normalize(paper_embed_1, p=2, dim=1)
    ebdmap[pid] = ebd
    return ebd

print(f'size {len(pdmap)}')
dfx = pd.read_csv('data/train_data.csv')
tokenizer, model = oagbert("oagbert-v2-sim")
model.eval()
bert = model.bert
cc = 0
with open('data/simdata.json','r') as jf:
    simMap = json.load(jf)
    print(f'start sim size {len(simMap)}')
    for i in tqdm(range(len(dfx))):
        pid = dfx.iloc[i]['id']
        refid = dfx.iloc[i]['ref_pid']
        if cc >= 20:
            break
        if pid in pdmap and refid in pdmap:
            if pid+refid not in simMap and refid+pid not in simMap:
                cc += 1
                try:
                    ebd1 = get_ebd(model,bert,pid)
                    ebd2 = get_ebd(model,bert,refid)
                except Exception as e:
                    print('Error: ',e)
                    continue
                sim = float(torch.mm(ebd1, ebd2.transpose(0, 1))[0][0])
                simMap[pid+refid] = sim
with open('data/simdata.json','r') as jf:
    simMap2 = json.load(jf)
    for pp in simMap:
        simMap2[pp] = simMap[pp]
print(f'end sim size {len(simMap2)}')
with open('data/simdata.json','w') as jf:
    json.dump(simMap2,jf)
