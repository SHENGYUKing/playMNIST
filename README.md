# 基于**PyTorch**框架和**MNIST**训练集的多种卷积神经网络模型的实现

## 使用数据
> **MNIST**<sup>[1]</sup>训练集    
> + Training set images: train-images-idx3-ubyte.gz (9,681 KB, 解压后 45,938 KB, 包含 60,000 个样本)    
> + Training set labels: train-labels-idx1-ubyte.gz (29 KB, 解压后 59 KB, 包含 60,000 个标签)    
> + Test set images: t10k-images-idx3-ubyte.gz (1,611 KB, 解压后 7,657 KB, 包含 10,000 个样本)    
> + Test set labels: t10k-labels-idx1-ubyte.gz (5 KB, 解压后 10 KB, 包含 10,000 个标签)    

## 训练环境
> 操作系统: Windows 10    
> 软件环境: Anaconda 3 + Python3.7        
> 使用框架: CUDA 8.0 + PyTorch 0.4.0    

## 使用算法
### I. **LeNet5**<sup>[2]</sup>    
> 一种最早用于数字识别的卷积神经网络(**Convolutional Neural Networks, CNN**<sup>[3]</sup>)。    
> 1. 每个卷积层实际包含三个部分，即卷积、池化和非线性激活函数。    
> 2. 利用卷积的方式提取空间特征。    
> 3. 采用降采样(Subsample)的平均池化层(Average Pooling)。    
> 4. 采用双曲正切(Tanh)或S型(Sigmoid)的非线性激活函数。    
> 5. 采用层与层之间的稀疏连接以减少参数数量从而减少计算复杂度。    
> 6. 采用多层感知机(Multilayer Perceptron, MLP)作为最后的分类器。    
---    
### II. **AlexNet**<sup>[4]</sup>    
> 2012年**ILSVRC(ImageNet Large Scale Visual Recognition Challenge)比赛**<sup>[5]</sup>冠军，比传统CNN更加迅速准确，相比LeNet以小而深的卷积层替代了大规模的卷积层。    
> 1. 采用ReLU函数作为CNN的激活函数，并验证了其在较深的神经网络效果优于S型函数，解决了S型函数在深层神经网络梯度弥散的问题，加快了训练速度。    
> 2. 采用Dropout正则化方法随机忽略部分神经元以避免模型过拟合，其主要应用于AlexNet最后部分的全连接层。    
> 3. 采用最大池化(Max Pooling)避免平均池化的模糊化效果，并且提出让步长比池化核的尺寸小从而使池化层的输出之间存在重叠和覆盖，提升了特征的丰富性。    
> 4. 采用局部响应归一化(Local Response Normalize, LRN)层对局部神经元的活动建立竞争机制，增益反馈较大的神经元同时抑制其他反馈较小的神经元，增强了模型泛化能力。    
> 5. 采用NVIDIA®CUDA™(Compute Unified Device Architecture)加速深度卷积神经网络的训练，使用GPU处理神经网络训练时大量的矩阵运算。    
> 6. 采用数据增强技术，在原始输入的256x256图像基础上随机地截取224x224大小的区域以及水平翻转的镜像，相当于增加了2x(256-224)^2=2048倍的数据量，由此减轻了因参数过多而导致的过拟合问题，增强了模型的泛化能力；利用模型进行预测时，提取图片的四角与中间并进行左右翻转从而共得到10张图片，对10张图片分别进行预测并求均值作为预测结果，提升了模型的准确率。    
---
### III. **ZFNet**<sup>[6]</sup>    
> 2013年ILSVRC比赛冠军，在AlexNet基础上进行微调，使用**ReLU激活函数**<sup>[7]</sup>与**交叉熵损失函数**<sup>[8]</sup>获得了更好的性能。    
> 1. 采用反卷积和可视化特征图(Feature Map)，并发现了特征的分层结构：浅层的网络主要提取到的是轮廓、边缘、颜色、纹理等特征，深层的网络提取到的是类别相关的抽象特征，由此说明越深层的特征其分类性能越好。    
> 2. 采用更小尺寸的卷积核和更小的步长代替了AlexNet在浅层卷积网络中的卷积核，从而保留了更多特征。    
> 3. 采用遮挡找出了决定图像类别的关键部位，并验证了深度的增加有利于网络提取更具区分意义的特征。    
> 4. 在AlexNet基础上进一步引入交叉熵损失函数，使得神经网络的分类准确性进一步提升。    
> 5. 验证了在网络训练时，浅层网络的参数收敛速度更快，随着层次的加深，模型收敛所需的训练时间将递增。  
---
### IV. **VGGNet**<sup>[9]</sup>    
> 2014年ILSVRC比赛TOP算法惜败于**GoogleNet**，在AlexNet基础上进一步加深网络层次以简化卷积神经网络结构。    
> 1. 删除了LRN层，随着神经网络的进一步加深，LRN层的作用被弱化几乎起不到应有的作用。    
> 2. 采用反复堆叠3x3的小型卷积核和2x2的最大池化层以进一步加深网络层次(16~19层)，通过不断加深网络结构以提升模型性能，使其分类识别错误率显著下降。    
> 3. 加深网络层次的同时简化了卷积神经网络的机构，但是训练时的特征数量变得非常庞大。    
---
### V. **GoogLeNet**<sup>[10]</sup>   
> 2014年ILSVRC比赛冠军，采用了独特的Inception模块以获得近似最优的局部稀疏解，另外本算法之所以被称为“GoogLeNet”而非“GoogleNet”是为了向“LeNet”致敬。    
> 1. 采用Inception模块，解决了随着网络深度加深导致的网络结构复杂化、参数增多、梯度消失等问题。    
> 2. 进一步加深网络(22层)，同时为了避免梯度消失，GoogLeNet在不同深度处增加了2个loss来保证梯度回传消失现象。    
> 3. 进一步拓宽特征图，采用多种卷积核(1x1, 3x3, 5x5)以及部分直接进行最大池化，并利用Inception模块压缩了特征图的厚度。    
> 4. 采用平均池化(Average Pooling)代替了全连接层，从而使参数数量相比AlexNet得到了显著减少，而整体性能更加优良。    
---
### VI. **ResNet**<sup>[11]</sup>    
> 2015年ILSVRC比赛冠军，引入残差(**Residual**<sup>[12]</sup>)概念，修正以往的卷积神经网络结构以适应更深层次的CNN训练。    
> 1. 引入恒等快捷连接(Identity Shortcut Connection)用于数据直接跨层处理，有效地在进一步加深网络的同时抑制了梯度消失和梯度爆炸，进一步提升了深度神经网络的性能。    
> 2. 利用残差模块将训练目标由H(x)转化为H(x)=F(x)+x中的F(x)，在不影响训练最终效果的基础上简化了训练难度，即将一个问题分解为多个尺度直接的残差问题从而起到优化训练的效果。    
---
### VII. **GAN**<sup>[13]</sup>  & **CGAN**<sup>[14]</sup>    
> 生成式对抗网络(**GAN**, **Generative Adversarial Networks**)是一种深度学习模型，是近年来最具前景的无监督学习方法之一。
> 1. GAN的任务有别于传统分类和识别，其主要能力是生成。    
> 2. GAN的主要结构包含两个模块，一个是判别模块(Generative Module)和生成模块(Generative Module)，从而可以根据已知训练集样本生成以假乱真的伪样本。    
> 3. 在GAN的基础之上衍生出了条件生成式对抗网络(**CGAN**, **Conditional Generative Adversarial Networks**)，从而满足生成特定要求的样本数据。
> 4. GAN和CGAN的出现一方面可以为生成式的任务提供全新的方法，另一方面也给训练集样本数量较少难以训练的项目提供了扩展训练样本的机会。

## 文件结构
```
|-- MNIST
    |-- mnist    // save the original dataset
    |   |-- t10k-images.idx3-ubyte
    |   |-- t10k-labels.idx1-ubyte
    |   |-- train-images.idx3-ubyte
    |   |-- train-labels.idx1-ubyte
    |   |-- gz
    |       |-- t10k-images-idx3-ubyte.gz
    |       |-- t10k-labels-idx1-ubyte.gz
    |       |-- train-images-idx3-ubyte.gz
    |       |-- train-labels-idx1-ubyte.gz
    |-- mnist_fake    // save the fake dataset by CGAN
    |   |-- fake_mnist.csv
    |-- maker.py    // CGAN
    |-- model.ckpt
    |-- model.py    // Load MNIST
    |-- README.md
    |-- train.py    // LeNet, AlexNet, ResNet...
    |-- img_gan
    |   |-- GPUvsCPU.txt
    |   |-- fake
    |   |   |-- byCPU
    |   |   |-- byGPU
    |   |-- real
    |   |   |-- byCPU
    |   |   |-- byGPU
    |-- model
    |   |-- cgan_discriminator_1E10.pth
    |   |-- cgan_generator_1E10.pth
    |   |-- cgan_discriminator_2E05.pth
    |   |-- cgan_generator_2E05.pth
```

## 相关文献与资料
> [1] [MNIST[EB/OL]. http://yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist)    
> [2][Lécun Y, Bottou L, Bengio Y, et al. Gradient-based learning applied to document recognition[J]. Proceedings of the IEEE, 1998, 86(11):2278-2324.](http://ieeexplore.ieee.org/iel4/5/15641/00726791.pdf)    
> [3][Convolutional Neural Network[EB/OL]. https://en.wikipedia.org/wiki/Convolutional_neural_network](https://en.wikipedia.org/wiki/Convolutional_neural_network)    
> [4][Krizhevsky A , Sutskever I , Hinton G . ImageNet Classification with Deep Convolutional Neural Networks[C]// NIPS. Curran Associates Inc. 2012.](http://www.researchgate.net/publication/267960550_ImageNe)    
> [5][LSVRC[EB/OL]. http://image-net.org/challenges/LSVRC](http://image-net.org/challenges/LSVRC)    
> [6][Zeiler M D , Fergus R . Visualizing and Understanding Convolutional Networks[M]// Computer Vision – ECCV 2014. Springer International Publishing, 2014.](http://link.springer.com/10.1007/978-3-319-10590-1_53)    
> [7][Glorot X , Bordes A , Bengio Y . Deep Sparse Rectifier Neural Networks[C]// Proceedings of the 14th International Conference on Artificial Intelligence and Statistics (AISTATS). 2010.](https://www.researchgate.net/publication/319770387_Deep_Sparse_Rectifier_Neural_Networks)    
> [8][James D. McCaffrey. Why You Should Use Cross-Entropy Error Instead Of Classification Error Or Mean Squared Error For Neural Network Classifier Training[EB/OL]. https://jamesmccaffrey.wordpress.com/2013/11/05/why-you-should-use-cross-entropy-error-instead-of-classification-error-or-mean-squared-error-for-neural-network-classifier-training/](https://jamesmccaffrey.wordpress.com/2013/11/05/why-you-should-use-cross-entropy-error-instead-of-classification-error-or-mean-squared-error-for-neural-network-classifier-training/)    
> [9][Simonyan, Karen, Zisserman, Andrew. Very Deep Convolutional Networks for Large-Scale Image Recognition[J]. Computer Science, 2014.](http://arxiv.org/abs/1409.1556)    
> [10][Szegedy C , Liu W , Jia Y , et al. Going Deeper with Convolutions[J]. 2014.](http://arxiv.org/abs/1409.4842)    
> [11][He K , Zhang X , Ren S , et al. Deep Residual Learning for Image Recognition[C]// 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, 2016.](http://arxiv.org/pdf/1512.03385)    
> [12][Residual[EB/OL]. https://en.wikipedia.org/wiki/Residual](https://en.wikipedia.org/wiki/Residual)    
> [13][Goodfellow I J , Pouget-Abadie J , Mirza M , et al. Generative Adversarial Networks[J]. Advances in Neural Information Processing Systems, 2014, 3:2672-2680.](https://arxiv.org/abs/1406.2661)    
> [14][Mehdi Mirza, Simon Osindero. Conditional Generative Adversarial Nets[J]. 2014.](https://arxiv.org/pdf/1411.1784.pdf)