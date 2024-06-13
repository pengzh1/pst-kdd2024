import pandas as pd
import json
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, StandardScaler, KBinsDiscretizer, \
    OneHotEncoder, MinMaxScaler
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU, Input, BatchNormalization
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight
from keras import models
from keras import layers
from keras import losses, metrics, optimizers, regularizers
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint, Callback
import time
import random
from sklearn.metrics import classification_report, precision_recall_fscore_support, average_precision_score


def kbdx(dft, col, bins=8):
    kbd = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
    kbd.fit(dft[dft['mpos'] != 2][[col]])
    dft[col] = kbd.transform(dft[[col]])


def prepare():
    # 1.合并数据，训练数据、验证集数据一起进行数据预处理
    with open('data/bert_fold6/train_submission_scibert_merge.json') as f:
        yscorebx = json.load(f)
    with open('data/bert_fold6/test_submission_scibert_merge.json') as f:
        yscorebx2 = json.load(f)
        for dd in yscorebx2:
            yscorebx[dd] = yscorebx2[dd]
    dfx = pd.read_csv('data/train_data.csv')
    # 文章引用数 - 需要构建特征间的非线性关系才可以用得到
    # dfx['recip_ref'] = dfx.apply(lambda row:  (79 if len(yscorebx[row['id']]) >=79 else len(yscorebx[row['id']]))//10,axis=1)
    # dfx['label'] = dfx.apply(lambda row:  1 if row['mpos']==1 and row['label'] == 0 else row['label'],axis=1)
    kbd = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    dfx['p_ref'] = dfx.apply(lambda row: len(yscorebx[row['id']]), axis=1)
    dfx['min_coref_cnt'] = dfx.apply(lambda row: min(row['min_coref_cnt'], 4), axis=1)
    kbdx(dfx, 'p_ref')
    kbdx(dfx, 'rev_refno')
    kbdx(dfx, 'bert_rank')
    kbdx(dfx, 'ref_rank')
    kbdx(dfx, 'cita_rank')
    kbdx(dfx, 'ref_no')
    kbdx(dfx, 'ref_ratio')
    kbdx(dfx, 'prank')
    kbdx(dfx, 'year_diff')
    kbdx(dfx, 'cita_cnt')
    kbdx(dfx, 'p_cita')
    dfx['common_authors'] = dfx.apply(lambda row: 1 if row['common_authors'] >= 1 else 0, axis=1)
    kmbd = KBinsDiscretizer(n_bins=6, encode='ordinal', strategy='kmeans')
    dfx['bert_score_cl'] = kmbd.fit_transform(dfx[['bert_score']])
    kbd2 = KBinsDiscretizer(n_bins=12, encode='ordinal', strategy='quantile')
    scaler = MinMaxScaler()
    print(max(dfx['pscore']), min(dfx['pscore']))
    dfx['pscore'] = scaler.fit_transform(dfx[['pscore']])
    dfx['hpid'] = dfx.apply(lambda row: 0 if row['ref_pid'] == '0' else 1, axis=1)
    # dfx['psim'] = dfx.apply(lambda row:int((100*row['bert_score']-60) // 2)+1,axis=1)
    print(dfx.dtypes)
    # dfx = dfx.drop(columns=['ref_pid'])
    return dfx
    # dfx['year_diff'] =  qt2.fit_transform(dfx[['year_diff']])


def get_input(dfx):
    categorical_features = ['abstract', 'introduction', 'relatedwork', 'conclusion', 'experiment', 'method',
                            'discussion', 'results', 'others',
                            'ref_no', 'ref_rank', 'common_authors',
                            'ref_ratio',
                            'cita_rank',
                            'cita_cnt',
                            'year_diff',
                            'prank',
                            'refone_cnt',
                            'hpid',
                            'rev_refno',
                            'min_coref_cnt', 'bert_score_cl',
                            'bert_rank',
                            'p_cita'
                            # 'p_ref'
                            # 'recip_ref'
                            ]
    num_features = [
        'bert_score', 'pscore', 'psim', 'sig_score', 'sigu_score'
    ]
    mtx = {}
    clen = 0
    dft = dfx
    for cf in categorical_features:
        enc = OneHotEncoder(handle_unknown='ignore')
        # enc.fit(dft[[cf]].values)
        matrix = enc.fit_transform(dft[[cf]].values).toarray()
        mtx[cf] = matrix
        print(f'{cf} col: {clen}--{clen + len(matrix[0]) - 1}')
        clen += len(matrix[0])
    for cf in num_features:
        clen += 1
        print(f'{cf} col: {clen}')

    print(clen)
    all_data = np.zeros((len(dft), clen))
    idx = 0
    for cf in mtx:
        for i in range(len(mtx[cf][0])):
            for j in range(len(all_data)):
                all_data[j][idx] = mtx[cf][j][i]
            idx += 1

    for nf in num_features:
        for i in range(len(all_data)):
            all_data[i][idx] = dft.loc[i, nf]
        idx += 1

    print(idx)
    all_data.shape
    all_label = np.array(dft['label'])
    return all_data, all_label, clen


def plot_metrics(history):
    metrics = ['loss', 'prc', 'auc', 'map']
    for n, metric in enumerate(metrics):
        name = metric
        plt.plot(history.epoch, history.history[metric], 'r', label='Train')
        plt.plot(history.epoch, history.history['val_' + metric], 'b', label='val')
        plt.xlabel('Epochs')
        plt.ylabel(name)
        plt.legend()
        plt.show()


class ROCAUCCallback(Callback):
    def __init__(self, filename, test_set=set(), fold=4):
        super(ROCAUCCallback, self).__init__()
        self.filename = filename
        self.best_maps = 0.0
        self.fold = fold
        self.test_set = test_set

    def map2_test(self, etrain=False):
        train_size = len(dfx[dfx['label'] > -1])
        test_data = []
        test_labels = []
        df_idx = []
        for i in range(train_size):
            if (i in self.test_set and not etrain) or (i not in self.test_set and etrain):
                test_data.append(all_data[i])
                test_labels.append(all_label[i])
                df_idx.append(i)
        test_data = np.array(test_data)
        test_labels = np.array(test_labels)
        # nd = mm.predict(test_data)
        pred = self.model.predict(test_data, batch_size=4096)
        nd = [it[0] for it in pred]
        cp = dfx.iloc[df_idx[0]]['id']
        cl = []
        cs = []
        mat = []
        for i in range(len(nd)):
            dfidx = df_idx[i]
            assert test_labels[i] == dfx.iloc[dfidx]['label']
            pid = dfx.iloc[dfidx]['id']
            if pid != cp:
                mat.append(average_precision_score(cl, cs))
                cp = pid
                cl = []
                cs = []
            if dfx.loc[dfidx, 'mpos'] == 2:
                cs.append(0.0)
                cl.append(0)
            else:
                cs.append(nd[i])
                cl.append(test_labels[i])
        mat.append(average_precision_score(cl, cs))
        return np.mean(mat)

    def on_epoch_end(self, epoch, logs={}):
        train_size = len(dfx[dfx['label'] > -1])
        fold = self.fold
        maps = self.map2_test()
        maps2 = self.map2_test(True)
        logs['val_map'] = maps
        logs['map'] = maps2
        if maps > self.best_maps:
            self.best_maps = maps
            if epoch > 1:
                self.model.save_weights(self.filename)
        print(f'\map: {maps:.4f} train map: {maps2:.4f}- best maps: {self.best_maps:.4f}', end=100 * ' ' + '\n')
        if (epoch + 1) % 5 == 0:
            print(f"{time.ctime()} Epoch {epoch + 1} completed.{logs}")


metriclist = [
    # metrics.BinaryCrossentropy(name='bce'),
    metrics.MeanSquaredError(name='mse'),
    metrics.FalsePositives(name='fp'),
    metrics.TruePositives(name='tp'),
    metrics.TrueNegatives(name='tn'),
    metrics.FalseNegatives(name='fn'),
    metrics.BinaryAccuracy(name='acc'),
    metrics.AUC(curve='PR', name='prc'),
    metrics.AUC(name='auc'),
    metrics.Recall(name='recall'),
    metrics.Precision(name='precision')
]

fmMap = {}


def create_model():
    model = Sequential()
    model.add(Input(shape=(clen,)))
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu',
                    kernel_regularizer=regularizers.l1(0.02)
                    ))
    # model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    rmsprop = optimizers.RMSprop(learning_rate=0.001)
    adam = optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=metriclist)
    return model


def create_fold_model(all_data, all_label, fold, clen, epoch=3):
    train_size = len(dfx[dfx['label'] > -1])
    test_idx = set()
    if fold > 4:
        all_train_pid = list(set(dfx[dfx['label'] > -1]['id'].unique()))
        test_pid = set(random.sample(all_train_pid, int(len(all_train_pid) / 4)))
        for i in range(train_size):
            if dfx.loc[i, 'id'] in test_pid:
                test_idx.add(i)
    else:
        test_start = int(train_size / 4) * (fold - 1)
        test_end = int(train_size / 4) * fold if fold < 4 else train_size
        test_idx = set([test_start + it for it in range(test_end - test_start)])
        print(f'fold{fold}：test {test_start}-{test_end}')
    # print(test_pid)
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for i in range(train_size):
        if dfx.loc[i, 'mpos'] == 2:
            continue
        dt = train_data
        lb = train_labels
        if i in test_idx:
            dt = test_data
            lb = test_labels
        dt.append(all_data[i])
        lb.append(all_label[i])
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    print(
        f'train pos {sum(train_labels)} test pos{sum(test_labels)}  data len {train_data.shape},{test_data.shape},{train_labels.shape},{test_labels.shape}')
    model = create_model()
    pos = sum(train_labels) + sum(test_labels)
    total = len(train_labels) + len(test_labels)
    neg = total - pos
    w0 = (1 / neg) * (total / 2.0)
    w1 = (1 / pos) * (total / 2.0)
    print(w0, w1)
    mm = model
    cw = {0.0: w0, 1.0: w1}
    history = mm.fit(train_data, train_labels, epochs=epoch, batch_size=64, validation_data=(test_data, test_labels),
                     class_weight=cw, verbose=0,
                     callbacks=[ROCAUCCallback(filename=f'data/dnnmodel/dnnfr{fold}.h5', test_set=test_idx, fold=fold)],
                     shuffle=True)
    plot_metrics(history)


def calc_avg(test_data, spec=-1):
    if spec > 0:
        model = create_model()
        model.load_weights(f'data/dnnmodel/dnnfr{spec}.h5')
        pred = model.predict(test_data)
        return [it[0] for it in pred]


def map_test(spec=-1):
    ixxs = [it for it in range(4)]
    if spec > 0 and spec < 5:
        ixxs = [spec - 1]
    for ix in ixxs:
        fold = ix + 1
        train_size = len(dfx[dfx['label'] > -1])
        test_start = int(train_size / 4) * (fold - 1)
        test_end = int(train_size / 4) * fold if fold < 4 else train_size
        print(f'fold{fold}：test {test_start}-{test_end}')
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        for i in range(train_size):
            dt = train_data
            lb = train_labels
            if i >= test_start and i < test_end:
                dt = test_data
                lb = test_labels
            dt.append(all_data[i])
            lb.append(all_label[i])
        train_data = np.array(train_data)
        test_data = np.array(test_data)
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)
        # nd = mm.predict(test_data)
        nd = calc_avg(test_data, spec)
        print(len(nd), sum(nd))

        cp = dfx.iloc[test_start]['id']
        cl = []
        cs = []
        cs2 = []
        mat = []
        mat2 = []
        resultf = {}
        for i in range(len(nd)):
            dfidx = test_start + i
            assert test_labels[i] == dfx.iloc[dfidx]['label']
            pid = dfx.iloc[dfidx]['id']
            if pid != cp:
                mat.append(average_precision_score(cl, cs))
                mat2.append(average_precision_score(cl, cs2))
                resultf[cp] = cs.copy()
                cp = pid
                cl = []
                cs = []
                cs2 = []
            if dfx.loc[dfidx, 'mpos'] == 2:
                cs.append(0.0)
                cs2.append(0.0)
                cl.append(0)
            else:
                cs.append(nd[i])
                cs2.append(dfx.iloc[dfidx]['bert_score'])
                cl.append(test_labels[i])
        mat.append(average_precision_score(cl, cs))
        mat2.append(average_precision_score(cl, cs2))
        resultf[cp] = cs.copy()
        print(f'dnn map {np.mean(mat):.4f}')
        print(f'dnn map {np.mean(mat2):.4f}')


def reference_test(all_data, dfx):
    otlen = len(dfx[dfx['label'] != -1])
    valid_data = all_data[otlen:]
    valid_label = all_label[otlen:]
    assert all_label[otlen] == -1 and all_label[otlen - 1] > -1
    bertfile = '/home/featurize/work/kddpst/6f01/test_submission_scibert_merge.json'
    with open(bertfile) as f:
        yscoreb = json.load(f)
    nd = calc_avg(valid_data, spec=5)
    print(len(nd), sum(nd))
    cp = dfx.iloc[otlen]['id']
    cs = []
    resultf = {}
    for i in range(len(nd)):
        dfidx = otlen + i
        pid = dfx.iloc[dfidx]['id']
        if pid != cp:
            resultf[cp] = cs.copy()
            assert len(cs) == len(yscoreb[cp])
            cp = pid
            cs = []
        if dfx.loc[dfidx, 'mpos'] == 2:
            cs.append(0)
        else:
            cs.append(float(nd[i]))
    resultf[cp] = cs.copy()
    assert len(cs) == len(yscoreb[cp])
    with open('dnntestsubmit.json', 'w', encoding='utf-8') as wf:
        json.dump(resultf, wf, indent=4, ensure_ascii=False)


dfx = prepare()
all_data, all_label, clen = get_input(dfx)
ne = 20
# train_model
# create_fold_model(all_data,all_label,fold=5,clen=clen,epoch=ne)
map_test(spec=5)
reference_test(all_data, dfx)