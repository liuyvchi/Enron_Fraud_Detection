__1. 向我们总结此项目的目标以及机器学习对于实现此目标有何帮助。作为答案的部分，提供一些数据集背景信息以及这些信息如何用于回答项目问题。你在获得数据时它们是否包含任何异常值，你是如何进行处理的？__

这个项目的目标是根据安然公司的邮件数据建立模型，提取特征，进行模式识别，预测有欺诈嫌疑的安然雇员。

数据集一共有146条数据。数据包含了14维的财务特征和6维的邮件特征（一共20个），以及代表POI标签的label信息。其中有18条records被标注为POI(Person of Interest),剩下的records被标注为非POI.

数据中有异常值，比如键为"LOCKHART EUGENE E"的这条记录它所有的fearture都为NaN，"THE TRAVEL AGENCY IN THE PARK"和"TOTAL"这两条记录不是单个雇员的记录。上面三条记录应该从数据集中清除掉

__2. 你最终在你的 POI 标识符中使用了什么特征，你使用了什么筛选过程来挑选它们？你是否需要进行任何缩放？为什么？作为任务的一部分，你应该尝试设计自己的特征，而非使用数据集中现成的——解释你尝试创建的特征及其基本原理。（你不一定要在最后的分析中使用它，而只设计并测试它）。在你的特征选择步骤，如果你使用了算法（如决策树），请也给出所使用特征的特征重要性；如果你使用了自动特征选择函数（如 SelectBest），请报告特征得分及你所选的参数值的原因。__

我通过`scikit-learn`里面的`SelectKBest`算法选取了5个表现最好的feature。它们分别是'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'deferred_income'.

需要进行特征缩放，因为有关finance的feature和有关email的feature在数据的量级上相差太大。为了公平地consider每个feature， 我用Scikit-learn的MinMaxScaler进行特征缩放。

我introduce了两个自定义的feature：`from_ratio`和`to_ratio`。 

`from_ratio = from_this_person_to_poi/from_messages`

`to_ratio = from_poi_to_this_person/to_messages` 

因为poi嫌疑人员应该会比非poi人员在与poi人员的邮件往来上更加的频繁，所以就用与poi人员往来邮件分别占总往来邮件的比例作为新的features。

下表是`SelectKBest`算法得出的排名前5的特征的得分(保留三位小数)：

| Selected Features       |  Score |
| :---------------------- | -----: |
| exercised_stock_options | 24.815 |
| total_stock_value       | 24.183 |
| bonus                   | 20.792 |
| salary                  | 18.290 |
| deferred_income         | 11.458 |

很遗憾，自己定义的特征没有被选进来。

__3. 你最终使用了什么算法？你还尝试了其他什么算法？不同算法之间的模型性能有何差异？__

我最终用了`GaussianNB`算法。 我还尝试了`SVM`，`RandomForest`，`AdaBoot`算法。

| algorithm    | accuracy_score | time(s)         |
| :----------- | -------------: | --------------- |
| GaussianNB   | 0.904761904762 | 0.0292749404907 |
| SVM          | 0.952380952381 | 0.100798130035  |
| RandomForest | 0.952380952381 | 69.0266699791   |
| AdaBoot      | 0.928571428571 | 8.74337697029   |

这里的accuracy是直接用`sklearn.metrics`里面分类器自带的`accuracy_score`方法算出来的，除了考虑`accuracy`和`time consumption`以外，还得考虑其他度量和评估因数。这将在下一个问题讨论。

__4. 调整算法的参数是什么意思，如果你不这样做会发生什么？你是如何调整特定算法的参数的？（一些算法没有需要调整的参数 – 如果你选择的算法是这种情况，指明并简要解释对于你最终未选择的模型或需要参数调整的不同模型，例如决策树分类器，你会怎么做)__

调整算法的参数指的是通过调整模型的参数使得模型能在测试集上有更好地预测结果。如果不合适的参数或者初始化，有可能会影响最终的performance, 比如会产生over fitting。 我是用`Scikit-learn`的`GridSearchCV`来自动选择参我提供的数中表现最好参数的，在数据集比较小的时候是比较适合的。这个思路是引用自[这里](https://github.com/haown1992/Enron/blob/master/poi_id.ipynb)。我需要做的就是提供相对diverse并且reseonable的候选参数。

__5. 什么是验证，未正确执行情况下的典型错误是什么？你是如何验证你的分析的？__

验证是测试训练的模型在验证集上的表现。经典错误是没有将验证集从训练集中分离出来导致过拟合。 

采用cross-validation(交叉验证)。因为此数据集的label不平衡,POI的records只有14条，所以用StratifiedShuffleSplit来确保training dataset和testing dataset能够被合理划分，即保持POI与非POI的比例尽可能相同，方法来自[这里](http://road2autodrive.info/2018/01/16/Uda-DataAnalysis-46-project/#分析报告从安然公司邮件中发现欺诈证据)。

__6. 给出至少 2 个评估度量并说明每个的平均性能。解释对用简单的语言表明算法性能的度量的解读。__

参见[这里](http://bookshadow.com/weblog/2014/06/10/precision-recall-f-measure/)

`Precision`:准确率 这个准确率指的是在我们预测为正的样本中，有多少比例的样本它们的ground truth也为真。

`recall`:召回率 召回率指的是在所有的预测了的样本中，那些本来的ground truth就为真的样本，有多少比例也是成功地被预测为真。

`F-Measure`：Precision和recall有时候是一对相对矛盾的指标，我们需要综合考虑。F-Measure是Precision和Recall加权调和平均。

公式：`F = (a**2 + 1)*P*R / a**2(P+R)` 当参数α=1时，就是`F1-score`。`F1-score`越高，说明模型的性能越好。

这个我用`tester.py`自带的`test_classifier`来对模型进行这两个量度的evaluation. 以下是结果：

| Selected Features | Precision | recall  | F1-score |
| :---------------- | --------: | ------- | -------- |
| GaussianNB        |   0.48876 | 0.38050 | 0.42789  |
| SVM               |   0.14530 | 0.00850 | 0.01606  |
| RandomForest      |   0.45536 | 0.12750 | 0.19922  |
| AdaBoot           |   0.36111 | 0.01300 | 0.02510  |

显然，GaussianNB变现最好。


#### reference
[推荐系统评测指标—准确率(Precision)、召回率(Recall)、F值(F-Measure)](http://bookshadow.com/weblog/2014/06/10/precision-recall-f-measure/)

[从安然公司邮件中发现欺诈证据](http://road2autodrive.info/2018/01/16/Uda-DataAnalysis-46-project/#分析报告从安然公司邮件中发现欺诈证据)

[Enron Fraud Analysis with Machine Learning](https://github.com/sagarnildass/Enron-Fraud-Analysis-with-Machine-Learning)

[Enron Fraud Detection](https://github.com/watanabe8760/udacity-da-p5-enron-fraud-detection)

[http://scikit-learn.org](http://scikit-learn.org/stable/)

