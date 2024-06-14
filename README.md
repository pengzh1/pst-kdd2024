# kdd-2024 paper-source-tracing edgeknife team work

We have made many attempts in different directions, so the  pipeline is  complex. Although most of the processing methods did not bring significant results, they are still listed below for reference

Some of the steps in the following pipeline are time-consuming, and some of the intermediate products and models have been stored on the [cloud storage](https://pan.baidu.com/s/16fVlq76UiXfuKkJlIimJlw?pwd=kte4)，code: kte4 ,

The work we do includes
- Using open-source project [SmartCite]（ https://github.com/pengzh1/SmartCite ）, and fix some bugs, parse the structure of the XML paper, see [smartcite. md](smartcite.md) for details
- Three BERT model scores were trained as sorting features using sentences that contain all references or only a single reference, as detailed in [bert.py](bert.py), [bert-sig.py](bert-sig.py), and [bert-sigunion.py](bert-sigunion.py)
- Using the page-rank algorithm, calculate the pagerank scores and rankings for each reference as sorting features
- Calculate whether each reference appears separately in a sentence as a sorting feature
- Use oagbert v2 sim to calculate the similarity score between the citation and the original text as the sorting feature. The calculation of simscore can be found in [sim.sh](sim.sh)
- Using a DNN neural network with 2 hidden layers, L1 regularization, and BatchNormalization to predict the final classification results



The process of preparing data features and training inference is as follows. Before executing the command, please download the data file and unzip it to the data directory
```shell
# need python 3.9
pip install -r requirements.txt
# prepare data
python date_prepare.py
# dnn train and inference
python dnn.py
```
