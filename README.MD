# 基于深度强化学习的文本情感分析
1. 本程序的模型基于tensorflow1.12下的tf.keras接口以及sklearn进行编写, 主要包括:
    - DQN模型; 
    - Bi-LSTM;
    - SVM.
2. 词向量的训练由gensim完成, 具体见run.py或dqn_sa.ipynb.


### 主要文件说明
- run.py 主执行代码, 包括模型训练评估;
- models.py 模型的定义;
- metrics.py 评估指标的定义, 包括accuracy, precision, recall, f1以及confusion matrix;
- dqn_sa.ipynb jupyter-notebook文件, 可在jupyter-notebook上直接打开并执行;
- requirements.txt 相关依赖;
- weibo_senti_100k 存放微博评论数据, 共119988条评论(正面:59993, 负面:59995);
- token 存放词向量文件夹;
- saved_model 存放训练完毕的模型.

### 运行
1. 首先运行脚本 pip install -r requirements.txt 安装依赖包
1. run.py为程序的启动代码, 运行脚本参考如下:
    - python run.py --model=DQN --mode=train
    - model为模型参数, 可选[DQN, LSTM, SVM]中的任一个;
    - mode为执行模式, 对DQN和LSTM有效, 可选[train, eval]中之一, train为训练, 
    eval为评估. 当使用评估模式时, 需要确保模型路径存在. SVM可忽略此项.
2. dqn_sa.ipynb的运行
    - 此处提供了ipynb脚本, 通过jupyter-notebook打开即可, 功能与第2点相同.

### 参考
https://zhuanlan.zhihu.com/p/32818105
