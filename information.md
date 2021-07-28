可以直接下载本项目，在自己的模型上进行微调。最好不要修改文件相对路径。在自己电脑上D盘下新建modelling文件夹，复制进去即可。

**Requirement:**

python 3.7

torch > 0.4.0

**Files Structure:**

`预训练文件：`

checkpoint:针对DenseNet + GC的权重文件；

checkpoint_classify:针对DenseNet + GC + SE的权重文件；

checkpoint_onlySE:针对DenseNet  + SE的权重文件；

checkpointpure:针对DenseNet的权重文件；

checkpoint_discrete:预测煤岩破坏剩余时间的权重文件。

`指标文件：`

files:针对DenseNet + GC的准确率、loss、混淆矩阵；

filespure:针对DenseNet的准确率、loss、混淆矩阵；

filesToSE:针对DenseNet + GC + SE的准确率、loss、混淆矩阵；

filesonlySE:针对DenseNet + SE的准确率、loss、混淆矩阵；

files_discrete:预测煤岩破坏剩余时间的准确率、loss、混淆矩阵。

`数据集：`

dataset_classify:分类数据存放目录；

dataset:回归数据存放目录。

`模型文件：`

model:模型构建文件；模型名字有明确的说明。

`process：数据预处理文件；`

`tools：绘图、保存准确率到json等工具文件；`

`trainModeler:训练文件；`

`config:配置文件；`

`eval:测试文件。`

`model_web:`

模型简易部署文件。配置好环境以后，只需点击.cmd 文件，打开localhost:5000。传入分类模型的路径。









