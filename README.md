# kdd-2024 paper-source-tracing edgeknife团队工作概述

我们做了很多不同方向的尝试，因此整体pipeline较为复杂，虽然其中大部分处理方法并没有带来明显的效果，但还是在下面列举出来以供参考

以下pipeline中部分步骤耗时较长，其中部分中间产物和模型已存放到百度网盘,链接地址:

我们所做的工作包括
- 使用开源项目 [SmartCite](https://github.com/pengzh1/SmartCite),并修复其中部分问题，解析xml论文结构,详见[smartcite.md](smartcite.md)
- 使用包含所有引用的句子、只包含单个引用的句子，训练了三个bert模型分数作为排序特征，详见[bert.py](bert.py) 、[bert-sig.py](bert-sig.py)、[bert-sigunion.py](bert-sigunion.py)
- 使用pagerank算法，计算各引用的pagerank分数及排名，作为排序特征
- 计算各个引用是否单独出现在某一句子中，作为排序特征
- 使用 oagbert-v2-sim计算出引用和原文相似度score作为排序特征，simscore计算详见 [sim.sh](sim.sh)
- 使用2层隐含层+L1正则化+BatchNormalization的DNN神经网络来预测最终分类结果



数据特征准备和训练推理过程如下，执行命令前请下载data文件并解压到data目录
```shell
pip install -r requirements.txt
# 准备数据及特征
python date_prepare.py
# dnn训练和推理
python dnn.py
```
