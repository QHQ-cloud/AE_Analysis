**Files Structure:**

checkpoint:针对DenseNet + GC的权重文件；

checkpoint_classify:针对DenseNet + GC + SE的权重文件；

checkpoint_onlySE:针对DenseNet  + SE的权重文件；

checkpointpure:针对DenseNet的权重文件；

files:针对DenseNet + GC的准确率、loss、混淆矩阵；

filespure:针对DenseNet的准确率、loss、混淆矩阵；

filesToSE:针对DenseNet + GC + SE的准确率、loss、混淆矩阵；

filesonlySE:针对DenseNet + SE的准确率、loss、混淆矩阵；

dataset_classify:数据存放目录；

model:模型构建文件；

process：数据预处理文件；

tools：绘图、保存准确率到json等工具文件；

trainModeler:训练文件；

config:配置文件；

eval:测试文件。



**Requirement:**

python 3.7

torch > 1.0.





