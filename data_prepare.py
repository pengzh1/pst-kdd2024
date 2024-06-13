import json
from os.path import join
import re
from tqdm import tqdm
from collections import OrderedDict
from collections import defaultdict as dd
import os
import random
from lxml import etree
from fuzzywuzzy import fuzz
import re
from bs4 import BeautifulSoup
import csv
from sknetwork.ranking import PageRank
import numpy as np
import sknetwork as skn
import pandas as pd


def load_json(rfdir, rfname):
    with open(join(rfdir, rfname), 'r', encoding='utf-8') as rf:
        data = json.load(rf)
        return data


def dump_json(obj, wfdir, wfname):
    with open(join(wfdir, wfname), 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, indent=4, ensure_ascii=False)


def handle_smartcite(pid):
    arf = {}
    patt = r'(sss(\s*,\s*sss)*){1,}'
    with open(f'/home/featurize/work/output/{pid}.xml.json', encoding='utf-8') as f:
        sc = json.load(f)
        textIdxMap = {}
        for para in sc['paragraphs']:
            for s in para:
                s['newtext'] = s['text']
                for rref in s['refTags']:
                    refidx = rref['reference_id']
                    text = rref['text']
                    match = re.search(r'\[[1-9][0-9]*\]', text)
                    if match and len(match.group(0)) >= 3:
                        textIdxMap[match.group(0)] = refidx
                    s['newtext'] = s['newtext'].replace(text, ' ')
        for para in sc['paragraphs']:
            for s in para:
                refedSet = set([item['reference_id'] for item in s['refTags']])
                for kk in textIdxMap:
                    if kk in s['newtext'] and textIdxMap[kk] not in refedSet:
                        refedSet.add(textIdxMap[kk])
                        s['newtext'] = s['newtext'].replace(kk, ' ')
                        s['refTags'].append({'reference_id': textIdxMap[kk], 'text': kk})
        for refid in sc['references']:
            ref = sc['references'][refid]
            match = re.search(r'(19|20)\d{2}', ref.get('year', ''))
            yearstr = ''
            if match:
                yearstr = match.group()
            arf[refid] = {
                'title': ref.get('article_title'),
                'year': yearstr,
                'source': ref.get('source'),
                'texts': [],
                'pHead': [],
                'refNum': []
            }
        ref_seq = {}
        for para in sc['paragraphs']:
            for s in para:
                for rref in s['refTags']:
                    oritxt = s['text']
                    # 替换目标论文为the paper
                    oritxt = oritxt.replace(rref['text'], 'the paper')
                    if rref['reference_id'] not in ref_seq:
                        ref_seq[rref['reference_id']] = len(ref_seq)
                    ref_no = ref_seq[rref['reference_id']]
                    # 替换其他论文为other paper
                    for rref2 in s['refTags']:
                        if rref2['reference_id'] != rref['reference_id']:
                            oritxt = oritxt.replace(rref2['text'], 'sss')
                    oritxt = re.sub(patt, 'other papers', oritxt)
                    oritxt = s['pHead'].lower() + ' ' + oritxt
                    arf[rref['reference_id']]['texts'].append(oritxt)
                    arf[rref['reference_id']]['pHead'].append(s['pHead'].lower())
                    arf[rref['reference_id']]['refno'] = ref_no
                    arf[rref['reference_id']]['refNum'].append(len(s['refTags']))
        arf = OrderedDict(sorted(arf.items(), key=lambda x: int(x[0][1:])))
        return arf


def merge_smartcite():
    data_dir = 'data/PST'
    with open(data_dir + '/' + 'paper_source_trace_train_ans.json', encoding='utf-8') as f:
        papers_train = json.load(f)
    with open(data_dir + '/' + 'paper_source_trace_valid_wo_ans.json', encoding='utf-8') as f:
        papers_valid = json.load(f)
    with open(data_dir + '/' + 'paper_source_trace_test_wo_ans.json', encoding='utf-8') as f:
        papers_test = json.load(f)
    pjson = {}
    for paper in tqdm(papers_train + papers_valid + papers_test):
        cur_pid = paper["_id"]
        pjson[cur_pid] = handle_smartcite(cur_pid)
    with open('data/' + 'smartcite2.json', mode='w', encoding='utf-8') as f:
        json.dump(pjson, f)


def splitDblp():
    data_dir = "data/PST"
    papers_train = load_json(data_dir, "paper_source_trace_train_ans.json")
    papers_valid = load_json(data_dir, "paper_source_trace_valid_wo_ans.json")
    papers_test = load_json(data_dir, "paper_source_trace_test_wo_ans.json")
    # dblp数据集，包含论文元信息及三层引文列表,
    pid_set = set()
    ref_set = set()
    ref_map = {}
    paperList = [papers_train, papers_valid, papers_test]
    psets = [set(), set(), set()]
    rsets = [set(), set(), set()]
    print(psets)
    for i in range(len(paperList)):
        paperds = paperList[i]
        for paper in paperds:
            cur_pid = paper["_id"]
            ref_map[cur_pid] = set()
            ref_ids = paper.get("references", [])
            pid_set.add(cur_pid)
            psets[i].add(cur_pid)
            for ref in ref_ids:
                ref_set.add(ref)
                ref_map[cur_pid].add(ref)
                rsets[i].add(ref)
            if 'refs_trace' in paper:
                for ref in paper['refs_trace']:
                    ref_set.add(ref['_id'])
                    ref_map[cur_pid].add(ref['_id'])
                    rsets[i].add(ref['_id'])
    print(len(pid_set), len(ref_set), [len(it) for it in psets], [len(it) for it in rsets])
    dblp_fname = "data/DBLP-Citation-network-V15.1.json"
    oag_fname = "data/oagmerge.json"
    dblpSet = set()
    # 遍历oag数据，补充pid-refid集合数据
    cnt = [0] * 3
    with open(oag_fname, "r", encoding="utf-8") as myFile:
        for i, line in tqdm(enumerate(myFile), desc="readDBLP", total=20000):
            if len(line) <= 10:
                continue
            paper_tmp = json.loads(line.strip())
            pid = paper_tmp["id"]
            if pid in pid_set:
                ref_ids = paper_tmp.get("references", [])
                for ref in ref_ids:
                    ref_set.add(ref)
                    ref_map[pid].add(ref)
                    for ixs, ss in enumerate(psets):
                        if pid in ss:
                            rsets[ixs].add(ref)
    print(len(pid_set), len(ref_set), [len(it) for it in psets], [len(it) for it in rsets])
    # 遍历dblp数据，开始写入新数据
    cntx1 = [0] * 3
    cntx2 = [0] * 3
    with open("/home/featurize/work/dblpmerge.json", "w", encoding="utf-8") as newFile:
        with open(dblp_fname, "r", encoding="utf-8") as myFile:
            for i, line in tqdm(enumerate(myFile), desc="readDBLP", total=6400000):
                if len(line) <= 10:
                    continue
                paper_tmp = json.loads(line.strip())
                if paper_tmp["id"] in pid_set:
                    paper_tmp['references'] = list(set(list(ref_map[paper_tmp['id']]) + paper_tmp['references']))
                    ref_ids = paper_tmp.get("references", [])
                    for ref in ref_ids:
                        ref_set.add(ref)
                        for ixs, ss in enumerate(psets):
                            if paper_tmp["id"] in ss:
                                rsets[ixs].add(ref)
                if paper_tmp["id"] in pid_set or paper_tmp["id"] in ref_set:
                    for ixs, ss in enumerate(psets):
                        if paper_tmp["id"] in ss:
                            cntx1[ixs] += 1
                    for ixs, ss in enumerate(rsets):
                        if paper_tmp["id"] in ss:
                            cntx2[ixs] += 1
                    newFile.write(json.dumps(paper_tmp) + "\n")
                    dblpSet.add(paper_tmp["id"])
        print(len(pid_set), len(ref_set), [len(it) for it in psets], [len(it) for it in rsets], cntx1, cntx2)
        with open(oag_fname, "r", encoding="utf-8") as myFile:
            for i, line in tqdm(enumerate(myFile), desc="readDBLP", total=20000):
                if len(line) <= 10:
                    continue
                paper_tmp = json.loads(line.strip())
                if (paper_tmp["id"] in pid_set or paper_tmp["id"] in ref_set) and paper_tmp["id"] not in dblpSet:
                    paper_tmp['references'] = list(
                        set(list(ref_map.get(paper_tmp['id'], set())) + paper_tmp['references']))
                    for ixs, ss in enumerate(psets):
                        if paper_tmp["id"] in ss:
                            cntx1[ixs] += 1
                    for ixs, ss in enumerate(rsets):
                        if paper_tmp["id"] in ss:
                            cntx2[ixs] += 1
                    newFile.write(json.dumps(paper_tmp) + "\n")
        print(len(pid_set), len(ref_set), [len(it) for it in psets], [len(it) for it in rsets], cntx1, cntx2)
        print(len(set(list(rsets[0]) + list(rsets[1] + list[rsets[2]]))))
    print(len(pid_set), len(ref_set))


def extractDblp():
    data_dir = "data/PST"
    # 训练集和验证集
    papers_train = load_json(data_dir, "paper_source_trace_train_ans.json")
    papers_test = load_json(data_dir, "paper_source_trace_test_wo_ans.json")
    # dblp数据集，包含论文元信息及引文列表
    paper_dict_open = {}
    dblp_fname = "/home/featurize/work/dblpmerge.json"
    with open(dblp_fname, "r", encoding="utf-8") as myFile:
        for i, line in tqdm(enumerate(myFile), total=30000):
            if len(line) <= 2:
                continue
            paper_tmp = json.loads(line.strip())
            paper_dict_open[paper_tmp["id"]] = paper_tmp
    for paper in tqdm(papers_train + papers_test, desc="extractDBLP"):
        cur_pid = paper["_id"]
        if cur_pid not in paper_dict_open:
            paper_dict_open[cur_pid] = {'id': paper['_id'], "n_citation": 1, 'title': paper['title'],
                                        'year': paper.get('year', 0), 'authors': paper['authors'],
                                        'references': paper['references'], 'keywords': []}
        if 'refs_trace' in paper:
            for ref in paper['refs_trace']:
                if ref['_id'] not in paper_dict_open:
                    paper_dict_open[ref['_id']] = {'id': ref['_id'], "n_citation": 1, 'title': ref['title'],
                                                   'year': ref.get('year', 0), 'authors': ref['authors'],
                                                   'references': [], 'keywords': []}
    paper_dict_hit = dd(dict)
    cntx = [0] * 3
    for paper in tqdm(papers_train + papers_test, desc="extractDBLP"):
        cur_pid = paper["_id"]
        ref_ids = paper.get("references", [])
        pids = set([cur_pid] + ref_ids)
        if cur_pid in paper_dict_open:
            pids = set(list(pids) + paper_dict_open[cur_pid]['references'])
        if 'refs_trace' in paper:
            for ref in paper['refs_trace']:
                pids.add(ref['_id'])
        for pid in pids:
            # 在dblp中找不到的跳过
            if pid not in paper_dict_open:
                continue
            # 获取所有作者姓名
            cur_paper_info = paper_dict_open[pid]
            cur_authors = [a.get("name", "") for a in cur_paper_info.get("authors", [])]
            # 获取被引用次数
            n_citation = cur_paper_info.get("n_citation", 0)
            # 标题
            title = cur_paper_info.get("title", "")
            paper_dict_hit[pid] = {"id": pid, "authors": cur_authors, "n_citation": n_citation, "title": title,
                                   'year': cur_paper_info.get('year', 0),
                                   'keywords': cur_paper_info.get('keywords', []),
                                   'abstract': cur_paper_info.get('abstract', ''),
                                   'venue': cur_paper_info.get('venue', ''),
                                   'references': cur_paper_info.get("references", [])}
    # for paper in tqdm(papers_train + papers_valid + papers_test, desc="extractDBLP"):
    #     cur_pid = paper["_id"]
    #     if cur_pid in paper_dict_hit:

    print("number of papers after filtering", len(paper_dict_hit))
    dump_json(paper_dict_hit, data_dir, "paper_info_hit_from_dblp.json")


def merge_bertscore():
    tc = {}
    with open('data/bert_fold6/test_submission_scibert_1.json', mode='r', encoding='utf-8') as f:
        tc = json.load(f)
    for ii in range(5):
        with open(f'data/bert_fold6/test_submission_scibert_{ii + 2}.json', mode='r', encoding='utf-8') as f:
            tvc = json.load(f)
            for dd in tvc:
                sr = tvc[dd]
                for i in range(len(sr)):
                    tc[dd][i] += sr[i]
    for dd in tc:
        sr = tc[dd]
        for i in range(len(sr)):
            sr[i] = sr[i] / 6
    with open('data/bert_fold6/test_submission_scibert_merge.json', mode='w', encoding='utf-8') as f:
        json.dump(tc, f)
    vc = {}
    lvalid = int(788 / 6)
    vcs = []
    for fo in range(6):
        with open(f'data/bert_fold6/train_submission_scibert_{fo + 1}.json', mode='r', encoding='utf-8') as f:
            vcs.append(json.load(f))
    papers = sorted(vcs[0].keys())
    for i, x in enumerate(papers):
        vc[x] = vcs[min(int(i / lvalid), 5)][x]
    with open('data/bert_fold6/train_submission_scibert_merge.json', mode='w', encoding='utf-8') as f:
        json.dump(vc, f)


def count_sim(a, b, rat):
    cnt = 0
    for xx in a:
        for yy in b:
            cur_sim = fuzz.ratio(xx, yy)
            if cur_sim >= rat:
                cnt += 1
                break
    return cnt


def extract_features():
    data_dir = "data/PST"
    # 训练数据
    with open(join(data_dir, "paper_source_trace_train_ans.json"), 'r', encoding='utf-8') as read_file:
        data_train_dic = json.load(read_file)
    all_train_id = [item['_id'] for item in data_train_dic]
    with open(join(data_dir, "paper_source_trace_test_wo_ans.json"), 'r', encoding='utf-8') as read_file:
        data_test_dic = json.load(read_file)
    all_test_id = [item['_id'] for item in data_test_dic]
    all_id = all_train_id + all_test_id
    data_dic = data_train_dic + data_test_dic
    data_list = []
    p_label_list = {}
    paper_info_more = load_json(data_dir, "paper_info_hit_from_dblp.json")
    print('originSize', len(paper_info_more))
    # 全量bert预测数据
    bertfile = 'data/bert_fold6/train_submission_scibert_merge.json'
    with open(bertfile) as f:
        yscore = json.load(f)
    bertfile = 'data/bert_fold6/test_submission_scibert_merge.json'
    with open(bertfile) as f:
        yscorev = json.load(f)
        for dd in yscorev:
            yscore[dd] = yscorev[dd]
    smartcite = load_json(data_dir, "smartcite2.json")
    submit_eg = {}
    cnt = 0
    cntft = 0
    for i, item in tqdm(enumerate(all_id), total=len(all_id)):
        # 含标签数据的论文元数据
        paper_dic = data_dic[i]
        paper_id = paper_dic['_id']
        is_train = paper_id in set(all_train_id)
        # print(paper_id)
        # 所有正例ID
        paper_positive_id = []
        paper_positive_set = set()
        paper_positive_titles = set()
        ref_ids_set = set(paper_dic.get("references", []))
        if paper_id in paper_info_more and 'references' in paper_info_more[paper_id]:
            for ref in paper_info_more[paper_id]['references']:
                ref_ids_set.add(ref)
        if is_train:
            paper_positive_id = [item['_id'] for item in paper_dic['refs_trace']]
            paper_positive_set = set()
            paper_positive_titles = set()
            for item in paper_dic['refs_trace']:
                paper_positive_titles.add(item['title'].lower())
                ref_ids_set.add(item['_id'])
                if 'referenced_serial_number' not in item:
                    cnt += 1
                else:
                    paper_positive_set.add(item['referenced_serial_number'])
        ref_ids = list(ref_ids_set)
        # 作者集合
        if paper_id not in paper_info_more:
            authors_set = set(item.get('name', '') for item in paper_dic.get('authors', []))
        else:
            authors_set = set(item for item in paper_info_more[paper_id].get('authors', []))
        # 引用列表
        try:
            path = f'data/PST/paper-xml/{paper_id}.xml'
            # path = f'/home/featurize/paper-xml/{paper_id}.xml'
            tree = etree.parse(path)
            root = tree.getroot()
            # 引用列表
            listBibl = root.xpath("//*[local-name()='listBibl']")[0]
            biblStruct = listBibl.getchildren()
            num_ref = len(biblStruct)
        except OSError:
            tree = None
            num_ref = 0
            print('not exits xml ' + paper_id)
        f = open(path, encoding='utf-8')
        xml = f.read()
        bs = BeautifulSoup(xml, "xml")
        f.close()

        # 引文标题和引文ID的映射
        cur_title_to_pid = {}
        total_ncitation = 0
        nct_arr = []
        for ref_id in ref_ids:
            if ref_id in paper_info_more:
                cur_title_to_pid[paper_info_more[ref_id]["title"].lower()] = ref_id
                total_ncitation += paper_info_more[ref_id]['n_citation']
                nct_arr.append(paper_info_more[ref_id]['n_citation'])
        references = bs.find_all("biblStruct")
        # 构建 引用标识(b2/b5...)与引用标题、引用标题和引用标识序号的映射
        bid_to_title = {}
        cur_title_to_b_idx = {}
        n_refs = 0
        for ref in references:
            if "xml:id" not in ref.attrs:
                continue
            bid = ref.attrs["xml:id"]
            b_idx = int(bid[1:]) + 1
            if ref.analytic is None or ref.analytic.title is None:
                continue
            else:
                if b_idx > n_refs:
                    n_refs = b_idx
            bid_to_title[bid] = ref.analytic.title.text.lower()
            cur_title_to_b_idx[ref.analytic.title.text.lower()] = b_idx - 1
        feature_map = []
        for j in range(n_refs):
            feature_map.append([0] * 23)
        total_ref = []
        for key in smartcite[paper_id]:
            total_ref.append(len(smartcite[paper_id][key]['texts']))
        assert n_refs == len(yscore[paper_id])
        for key in smartcite[paper_id]:
            if smartcite[paper_id][key]['title'] != None and len(smartcite[paper_id][key].get('title', '')) > 1:
                if key not in bid_to_title and int(key[1:]) < n_refs:
                    cntft += 1
                    bid_to_title[key] = smartcite[paper_id][key].get('title', '').lower()
                    cur_title_to_b_idx[smartcite[paper_id][key].get('title', '').lower()] = int(key[1:])
                    feature_map[int(key[1:])][21] = 2
        ref_sum = sum(total_ref)
        sortscore = sorted(yscore[paper_id])
        sortcitation = sorted(nct_arr)
        sortref = sorted(total_ref)
        pyear = paper_dic.get('year', 0)
        for key in smartcite[paper_id]:
            scc = smartcite[paper_id][key]
            idx = int(key[1:])
            if idx >= n_refs:
                continue
            feature_map[idx][20] = 1
            if paper_id in paper_info_more:
                feature_map[idx][20] = paper_info_more[paper_id].get('n_citation', 1)
            # 0 原文中引文被引次数/原文所有引用次数 占比
            feature_map[idx][0] = len(scc['texts']) / ref_sum
            # 1 原文中引文被引次数排名
            feature_map[idx][1] = (len(sortref) - sortref.index(len(scc['texts'])))
            # 2 引文语义分类预测值
            feature_map[idx][2] = yscore[paper_id][idx]
            # 3 引文语义分类预测值排名
            feature_map[idx][3] = (len(sortscore) - sortscore.index(yscore[paper_id][idx]))
            # 4 引文序号
            feature_map[idx][4] = scc.get('refno', len(sortref))
            # 5 年份差值
            if len(scc['year']) > 1 and pyear > 0:
                diff = paper_dic['year'] - int(scc['year'])
                if diff >= 0:
                    feature_map[idx][5] = min(diff, 20)
            else:
                feature_map[idx][5] = 7
            # 6 - 14 引用出现的段落，
            # 6 abstract 7 introduction/motivation 8 related work/background/preliminaries
            # 9 experiment/evaluation 10 conclusion 11 method/approach 12 discussion 13 results
            # 14 others
            phMap = {'abstract': 6, 'introduction': 7, 'motivation': 7, 'related work': 8, 'background': 8,
                     'preliminaries': 8, 'experiment': 9, 'evaluation': 9, 'conclusion': 10, 'method': 11,
                     'approach': 11, 'discussion': 12, 'results': 13
                     }
            for found_text in scc['pHead']:
                maxScore = 0
                maxIdx = 14
                for kk in phMap:
                    sscore = fuzz.partial_ratio(kk, found_text)
                    if sscore >= 65 and maxScore < sscore:
                        maxScore = sscore
                        maxIdx = phMap[kk]
                feature_map[idx][maxIdx] = 1
            feature_map[idx][16] = n_refs
            feature_map[idx][22] = n_refs - idx
        if is_train:
            label_list = [0] * n_refs
        else:
            label_list = [-1] * n_refs
        if is_train:
            for tt in cur_title_to_b_idx:
                for ptt in paper_positive_titles:
                    cur_sim = fuzz.ratio(tt, ptt)
                    if cur_sim >= 80 and cur_title_to_b_idx[tt] < n_refs:
                        label_list[cur_title_to_b_idx[tt]] = 1
        keywords = paper_info_more.get(paper_id, {}).get('keywords', [])
        for r_idx, ref_id in enumerate(ref_ids):
            if ref_id not in paper_info_more:
                continue
            cur_title = paper_info_more[ref_id].get("title", "").lower()
            if len(cur_title) == 0:
                continue
            cur_b_idx = None
            for b_title in cur_title_to_b_idx:
                cur_sim = fuzz.ratio(cur_title, b_title)
                if cur_sim >= 80:
                    cur_b_idx = cur_title_to_b_idx[b_title]
                    break
            if cur_b_idx is None:
                continue
            if cur_b_idx > n_refs:
                continue

            if ref_id in paper_positive_id and cur_b_idx not in paper_positive_set:
                feature_map[cur_b_idx][21] = 1
            if int(paper_info_more[ref_id].get("year", 0)) > 0 and pyear > 0:
                feature_map[cur_b_idx][5] = min(pyear - int(paper_info_more[ref_id].get("year", 0)), 20)
            # 找到了引用的序号，开始处理特征
            query_result = paper_info_more.get(ref_id, {})
            # 15: 引文在所有论文中的引用数
            feature_map[cur_b_idx][15] = query_result.get('n_citation', 0)
            # 16: 引文在所有论文中的引用数在本文引文中的排名
            feature_map[cur_b_idx][16] = len(sortcitation) - sortcitation.index(query_result.get('n_citation', 0))
            # 17: 引文是否与原文有重叠作者，0无1有
            ref_authors_set = set(query_result.get('authors', []))
            feature_map[cur_b_idx][17] = count_sim(ref_authors_set, authors_set, 88)
            # 18: 引文与原文重叠关键词个数
            ref_keywords = set(query_result.get('keywords', []))
            # print(ref_keywords,'----',keywords)
            feature_map[cur_b_idx][18] = count_sim(ref_keywords, keywords, 70)
            # 19: 引文ID
            feature_map[cur_b_idx][19] = ref_id
        # if is_train:
        #     for iii in paper_positive_set:
        #         label_list[iii] = 1
        p_label_list[paper_id] = label_list
        for i in range(len(feature_map)):
            data_list.append([paper_id, i] + feature_map[i] + [label_list[i]])
    # return
    out_dir = join('data')
    print('unrecog ', cnt, ' fillcnt', cntft)
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir + '/train_data.csv', 'w') as td:
        writer = csv.writer(td)
        writer.writerow(['id', 'ref_idx', 'ref_ratio', 'ref_rank', 'bert_score', 'bert_rank',
                         'ref_no', 'year_diff', 'abstract', 'introduction', 'relatedwork',
                         'experiment', 'conclusion', 'method', 'discussion', 'results', 'others', 'cita_cnt',
                         'cita_rank', 'common_authors', 'common_keyword', 'ref_pid', 'p_cita', 'mpos', 'rev_refno',
                         'label'
                         ])
        writer.writerows(data_list)


def oagbertsim():
    import pandas as pd
    dfx = pd.read_csv('data/train_data.csv')
    if 'psim' not in dfx.columns:
        dfx['psim'] = 0.0
    with open('data/simdata.json', 'r') as f:
        simMap = json.load(f)
    print('simSize', len(dfx[dfx['psim'] > 0]))
    for i in tqdm(range(len(dfx))):
        pid = dfx.loc[i, 'id'] + dfx.loc[i, 'ref_pid']
        pid2 = dfx.loc[i, 'ref_pid'] + dfx.loc[i, 'id']
        if pid in simMap:
            dfx.loc[i, 'psim'] = simMap[pid]
        if pid2 in simMap:
            dfx.loc[i, 'psim'] = simMap[pid2]
    print('simSize', len(dfx[dfx['psim'] > 0]))
    dfx.to_csv('data/train_data.csv', index=False)


def computePrank(pid, scoreMapMap, refIdMap):
    if pid in scoreMapMap:
        return scoreMapMap[pid]
    if pid not in refIdMap or len(refIdMap[pid]) == 0:
        return {}
    idMap = {}
    revMap = {}
    edges = []
    idMap[pid] = len(idMap)
    revMap[idMap[pid]] = pid
    # 两层依赖
    for ref in refIdMap[pid]:
        if ref not in idMap:
            idMap[ref] = len(idMap)
            revMap[idMap[ref]] = ref
        edges.append((idMap[pid], idMap[ref]))
        for ref2 in refIdMap.get(ref, []):
            if ref2 not in idMap:
                idMap[ref2] = len(idMap)
                revMap[idMap[ref2]] = ref2
            edges.append((idMap[ref], idMap[ref2]))
    seeds = {0: len(idMap)}
    pagerank = PageRank()

    g = skn.data.from_edge_list(edges, directed=True)
    scorer = pagerank.fit_predict(g, seeds)

    refSet = set(refIdMap[pid])
    sss = []
    for i in range(len(scorer)):
        if revMap[i] in refSet:
            sss.append(scorer[i])
    sumcnt = len(sss)
    scoreMap = {}
    scorearr = []
    for i in range(len(scorer)):
        if revMap[i] in refSet:
            sct = scorer[i] * sumcnt
            scorearr.append(sct)
            scoreMap[revMap[i]] = {'score': sct}
    scorearr = sorted(scorearr)
    for kk in scoreMap:
        scoreMap[kk]['rank'] = len(scorearr) - scorearr.index(scoreMap[kk]['score'])
    scoreMapMap[pid] = scoreMap
    return scoreMap


def compute_pagerank():
    refIdMap = {}
    scoreMapMap = {}
    dblp_fname = "data/dblpmerge.json"
    with open(dblp_fname, "r", encoding="utf-8") as myFile:
        for i, line in tqdm(enumerate(myFile), total=2000000):
            if len(line) <= 2:
                continue
            paper_tmp = json.loads(line.strip())
            refIdMap[paper_tmp["id"]] = paper_tmp.get("references", [])
    dfx = pd.read_csv('data/train_data.csv')
    dfx['prank'] = 0
    dfx['pscore'] = 0.0
    for i in tqdm(range(len(dfx))):
        pid = dfx.loc[i, 'id']
        prank = computePrank(pid, scoreMapMap, refIdMap)
        refpid = dfx.loc[i, 'ref_pid']
        if refpid in prank:
            dfx.loc[i, 'prank'] = prank[refpid]['rank']
            dfx.loc[i, 'pscore'] = prank[refpid]['score'] if prank[refpid]['score'] <= 1.0 else 1.0
        else:
            dfx.loc[i, 'prank'] = len(prank)
    dfx.to_csv('data/train_data.csv', index=False)


def fill_refcnt():
    dfx = pd.read_csv('data/train_data.csv')
    with open('data/smartcite2.json') as f:
        sc = json.load(f)

    dfx['refone_cnt'] = 0
    for i in tqdm(range(len(dfx))):
        pid = dfx.loc[i, 'id']
        ref_idx = dfx.loc[i, 'ref_idx']
        ref = sc[pid][f'b{ref_idx}']
        cnt = 0
        min_cnt = 10
        for num in ref['refNum']:
            if num == 1:
                cnt += 1
            if num < min_cnt:
                min_cnt = num
        dfx.loc[i, 'refone_cnt'] = 3 if cnt > 3 else cnt
        dfx.loc[i, 'min_coref_cnt'] = min_cnt
    dfx.to_csv('data/train_data.csv', index=False)


def merge_sigbertscore():
    tc = {}
    with open('data/sigbert_fold4/test_sig_bert_1.json', mode='r', encoding='utf-8') as f:
        tc = json.load(f)
    for ii in range(3):
        with open(f'data/sigbert_fold4/test_sig_bert_{ii + 2}.json', mode='r', encoding='utf-8') as f:
            tvc = json.load(f)
            for dd in tvc:
                sr = tvc[dd]
                for ik in sr:
                    arr = sr[ik]
                    for iii in range(len(arr)):
                        tc[dd][ik][iii] += arr[iii]
    sigScoreMap = {}
    for dd in tc:
        sr = tc[dd]
        for ik in sr:
            for ii in range(len(sr[ik])):
                sr[ik][ii] /= 4
            sigScoreMap[dd + ik] = max(sr[ik]) if len(sr[ik]) > 0 else 0.0
    lvalid = int(788 / 4)
    vcs = []
    for fo in range(4):
        with open(f'data/sigbert_fold4/train_sig_bert_{fo + 1}.json', mode='r', encoding='utf-8') as f:
            vcs.append(json.load(f))
    papers = sorted(vcs[0].keys())
    for i, x in enumerate(papers):
        sr = vcs[min(int(i / lvalid), 3)][x]
        for ik in sr:
            sigScoreMap[x + ik] = max(sr[ik]) if len(sr[ik]) > 0 else 0.0
    print('ready', len(sigScoreMap))
    dfx = pd.read_csv('data/train_data.csv')
    dfx['sig_score'] = 0.0
    for i in tqdm(range(len(dfx))):
        pid = dfx.loc[i, 'id']
        ref_idx = dfx.loc[i, 'ref_idx']
        if pid + str(ref_idx) in sigScoreMap:
            dfx.loc[i, 'sig_score'] = sigScoreMap[pid + str(ref_idx)]
    dfx.to_csv('data/train_data.csv', index=False)


def merge_sigbert_union_score():
    tc = {}
    sigUnionScoreMap = {}
    for ii in range(4):
        with open(f'data/sigunionbert_fold4/test_sig_union_bert_{ii + 1}.json', mode='r', encoding='utf-8') as f:
            tvc = json.load(f)
            for dd in tvc:
                sr = tvc[dd]
                for ik in sr:
                    sigUnionScoreMap[dd + ik] = sigUnionScoreMap.get(dd + ik, 0.0) + sr[ik]
    for kk in sigUnionScoreMap:
        sigUnionScoreMap[kk] /= 4
    print('ready', len(sigUnionScoreMap))
    lvalid = int(788 / 4)
    vcs = []
    for fo in range(4):
        with open(f'data/sigunionbert_fold4/train_sig_union_bert_{fo + 1}.json', mode='r', encoding='utf-8') as f:
            vcs.append(json.load(f))
    papers = sorted(vcs[0].keys())
    for i, x in enumerate(papers):
        sr = vcs[min(int(i / lvalid), 3)][x]
        for ik in sr:
            sigUnionScoreMap[x + ik] = sr[ik]
    print('ready', len(sigUnionScoreMap))
    dfx = pd.read_csv('data/train_data.csv')
    dfx['sigu_score'] = 0.0
    for i in tqdm(range(len(dfx))):
        pid = dfx.loc[i, 'id']
        ref_idx = dfx.loc[i, 'ref_idx']
        if pid + str(ref_idx) in sigUnionScoreMap:
            dfx.loc[i, 'sigu_score'] = sigUnionScoreMap[pid + str(ref_idx)]
    dfx.to_csv('data/train_data.csv', index=False)


# 合并处理smartcite产出的论文结构json
print('----------start merge smartcite json')
merge_smartcite()
print('----------end merge smartcite json')

# 抽取dblp及oag元数据信息
print('----------start extract dblp and oag data')
# splitDblp()
extractDblp()
print('----------end extract dblp and oag data')

# 合并通过6折训练的bert结果分数，其中test集使用6个模型分数的均值，train使用6个模型各自验证集分数合并
print('----------start merge bert fold6 score ')
merge_bertscore()
print('----------end merge bert fold6 score')

# 解析论文xml,生成特征csv
print('----------start extract paper features ')
extract_features()
print('----------end extract paper features ')

# 合并使用 oagbert-v2-sim计算出的论文score到traindata.csv中，simscore计算详见 sim.sh/sim.py
print('----------start merge oag-bert-sim score ')
oagbertsim()
print('----------end merge oag-bert-sim score ')

# 使用pagerank算法，计算各应用的pagerank分数及排名
print('----------start merge pagerank score ')
compute_pagerank()
print('----------end merge pagerank score ')

# 计算各个引用是否单独出现在某一句子中
print('----------start ref cnt stat ')
fill_refcnt()
print('----------end ref cnt stat ')

# 添加引用单独出现的句子的bertscore，训练脚本分别是 bertsig.py和bertsig-union.py
print('----------start merge bert sig score')
merge_sigbertscore()
merge_sigbert_union_score()
print('----------end merge bert sig score')