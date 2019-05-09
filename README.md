#  Model compression via distillation and quantization

This code has been written to experiment with quantized distillation and differentiable quantization, techniques developed in our paper ["Model compression via distillation and quantization"](https://arxiv.org/abs/1802.05668).

If you find this code useful in your research, please cite the paper:

```
@article{2018arXiv180205668P,
   author = {{Polino}, A. and {Pascanu}, R. and {Alistarh}, D.},
    title = "{Model compression via distillation and quantization}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1802.05668},
 keywords = {Computer Science - Neural and Evolutionary Computing, Computer Science - Learning},
     year = 2018,
    month = feb,
}
```


The code is written in [Pytorch 0.3](http://pytorch.org/) using Python 3.6. It is not backward compatible with Python2.x

*Note* Pytorch 0.4 introduced some major breaking changes. To use this code, please use Pytorch 0.3.

Check for the compatible version of torchvision. To run the code, use torchvision 0.2.0.
```
pip install torchvision==0.2.0
```
This should be done after installing the [requirements](requirements.txt).

# Getting started

### Prerequisites
This code is mostly self contained. Only a few additional libraries are requires, specified in [requirements.txt](requirements.txt). The repository already contains a fork of the [openNMT-py project](https://github.com/OpenNMT/OpenNMT-py). Note that, due to the rapidly changing nature of the openNMT-py codebase and the substantial time and effort required to make it compatible with our code, it is unlikely that we will support newer versions of openNMT-py.

### 文件内容
文件夹内容简要介绍如下:
 - *datasets* 该包用于自动下载以及预处理一些数据集, 包括CIFAR10, PennTreeBank, WMT2013等.
 - *quantization* 包含所使用的量化函数.
 - *perl_scripts* 包含一些perl脚本,来自于 [moses project](https://github.com/moses-smt/mosesdecoder) ,用于翻译任务.
 - *onmt*  代码来自于[openNMT-py project](https://github.com/OpenNMT/OpenNMT-py). 进行了少量修改,以使其与本项目代码兼容.
 - *helpers* 包含一些在整个工程中都会用到的函数.
 - *model_manager.py* 包含一些在保存模型时常用的I/O类. 特别在对多个相似的模型进行训练时有用, 对模型训练的选项进行追踪,并追踪训练时的结果.*Note*: 不支持同时对同一文件进行处理. I am working on a version that does; if you are interested, drop me a line.
 - 根目录下的文件,如 [cifar10_test.py](cifar10_test.py) 等,是主要文件,使用其他部分的代码完成实验.
 - 其他文件包含模型的定义,训练流程等.

### 运行代码
首先导入一些数据集,同时创建训练,测试集加载器.
创建一个存放所有数据集的文件夹；数据集将被自动下载,并在指定的文件夹中进行被处理.
下列代码展示如何加载CIFAR10数据集,并创建和训练模型.

```python
import datasets
datasets.BASE_DATA_FOLDER = '/home/saved_datasets'

batch_size = 50
cifar10 = datasets.CIFAR10() #-> will be saved in /home/saved_datasets/cifar10
train_loader, test_loader = cifar10.getTrainLoader(batch_size), cifar10.getTestLoader(batch_size)
```
接着,我们可以使用```train_loader```和```test_loader```来生成训练和测试数据.

此时,我们只需要定义一个模型并对其进行训练:

```python
import os
import cnn_models.conv_forward_model as convForwModel
import cnn_models.help_fun as cnn_hf
teacherModel = convForwModel.ConvolForwardNet(**convForwModel.teacherModelSpec,
                                              useBatchNorm=True,
                                              useAffineTransformInBatchNorm=True)
convForwModel.train_model(teacherModel, train_loader, test_loader, epochs_to_train=200)
```

正如之前所说的,最好使用ModelManager类别来自动保存和加载结果.所以代码一般具有如下形式:

```python
import os
import cnn_models.conv_forward_model as convForwModel
import cnn_models.help_fun as cnn_hf
import model_manager
cifar10Manager = model_manager.ModelManager('model_manager_cifar10.tst',
                                            'model_manager', create_new_model_manager=False)#the first time set this to True
model_name = 'cifar10_teacher'
cifar10modelsFolder = '~/quantized_distillation/'
teacherModelPath = os.path.join(cifar10modelsFolder, model_name)
teacherModel = convForwModel.ConvolForwardNet(**convForwModel.teacherModelSpec,
                                              useBatchNorm=True,
                                              useAffineTransformInBatchNorm=True)
if not model_name in cifar10Manager.saved_models:
    cifar10Manager.add_new_model(model_name, teacherModelPath,
            arguments_creator_function={**convForwModel.teacherModelSpec,
                                        'useBatchNorm':True,
                                        'useAffineTransformInBatchNorm':True})
cifar10Manager.train_model(teacherModel, model_name=model_name,
                           train_function=convForwModel.train_model,
                           arguments_train_function={'epochs_to_train': 200},
                           train_loader=train_loader, test_loader=test_loader)
```         
上述内容就是使用代码的一般结构,更多内容参考主文件. 

# Authors

 - Antonio Polino
 - Razvan Pascanu
 - Dan Alistarh

# License

The code is licensed under the MIT Licence. See the [LICENSE.md](LICENSE.md) file for detail.

# Acknowledgements

We would like to thank Ce Zhang  (ETH Zürich), Hantian Zhang (ETH Zürich) and Martin Jaggi (EPFL) for their support with experiments and valuable feedback.
