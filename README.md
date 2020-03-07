# Trigger Tasks Report GSoC 2020

This is the technicle report of the evaluation task of the GSoC 2020 project Deep Learning Algorithms for Muon Momentum Estimation in the CMS Trigger System provided by CMS at CERN.

In this task, I 
1. implmented Fully Connect Network (FCN), Convolutional Neural Network (CNN, ResNet, specifally) and a simple linear regression model to predict the energy level of [given particle dataset](https://www.dropbox.com/s/c1pzdacnzhvi6pm/histos_tba.20.npz?dl=0);
2. then reconstructed the feature of particle into a image-like 2D array and, again, fed as dataset to train ResNet;
3. implmented and testing Message Passing Neural Network (MPNN) in a quark/gulon classifier.

## Task 1

### Dataset Prepartion

The [given dataset](https://www.dropbox.com/s/c1pzdacnzhvi6pm/histos_tba.20.npz?dl=0) contains sparse feature matrix including multiple NaN values, so I first set them to `0` and normalized the feature matrix into a `mean=0, std=1` matrix. You can find it in task1.ipynb ### Explore Dataset and Standardization.

Also, the `q/pt` coloum in the `parameter` indicates the **momentum of particle**, which can be calcualte from `pt=1/(q/pt)` (since `q=+-1`). And thus I yeild the momentum of particle (as well as their classes of energy). Noteworthy, the eta and phi angle coloums are dropped.

Meanwhile, during the exploration, I find that the dataset is extremaly unbalance - up to 97% of particle has an energy of `0-10GeV`. I upsampled the remaining 3% dataset (by dupulate them simply) and drop some of the class zero (`0-10GeV`) to control the size of the whole dataset. Finally, the whole dataset contains 4 million particles, with around 1 million in each class. You can find it in task1.ipynb ### Upsampling Biased Dataset.

### Dataloader and Dataset

I implmented a `MuonDataset` and a batch-based dataloader of the `MuonDataset`. You can find it in task1.ipynb ### Data Loader. So, the comsumption of RAM in training process is reduced. 

The feature of a partcle is a `[87, 1]`array in `tensor.float`,while the label is a sclar (`0, 1, 2, 3`) in `tensor.long` . The training set includes `3000000` particles and the test set has `957571`particles.

### Fully Connected Net

At the very beginning, I implented a simple FCN as the base line model. It contains 3 linear layers followed by batch normalization layers. `ReLU` is served as the activation function.  The structure of my FCN is shown as followed.

```
FCNetwork(
  (fc1): Linear(in_features=87, out_features=100, bias=True)
  (bn1): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=100, out_features=100, bias=True)
  (bn2): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc3): Linear(in_features=100, out_features=100, bias=True)
  (bn3): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (predict): Linear(in_features=100, out_features=4, bias=True)
)
```

I choose `SGD` as the optimizer and   `CrossEntropyLoss` as the loss function. A more detailed hyperparameter list is shown.

```
num_epochs = 10
batch_size = 256
lr = 0.1
device='cpu'
lossFunction = nn.CrossEntropyLoss()
model = FCNetwork(87, 100, 4)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
```

The FCN classifier reached an accuracy of 51.584% in test set, with a loss at `1.189`. The train/test loss curves and the outout of the last epoch are shown. The y-axis value should be divided by the batch size. 

```
Epoch: 9
Train loss: 1.201 | Train Acc: 49.950% (1498494/3000000)
Test Loss: 1.189  | Test Acc: 51.584% (493958/957571)
```

![fcn1](asserts\fcn1.png)

![fcn2](asserts\fcn2.png)

### CNN

I implmented a ResNet in this part.  The detail of my implmentation is shown in task1.ipynb ### CNN. The CNN contains residual block with two `3*3` conv layers. And this ResNet involves three layers with two residual block (ResNet-16). The detail information of the net is shown. Noteworthy, I reconstructed the feature array into a 2D array.

```
ResNet(
  (conv): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (layer1): Sequential(
    (0): ResidualBlock(
      (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): ResidualBlock(
      (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer2): Sequential(
    (0): ResidualBlock(
      (conv1): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): ResidualBlock(
      (conv1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer3): Sequential(
    (0): ResidualBlock(
      (conv1): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): ResidualBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avg_pool): AvgPool2d(kernel_size=8, stride=8, padding=0)
  (fc): Linear(in_features=64, out_features=4, bias=True)
)
```

I choose `Adam` as the optimizer and   `CrossEntropyLoss` as the loss function. A more detailed hyperparameter list is shown.

```
batch_size = 256
num_epochs = 10
lossFunction = nn.CrossEntropyLoss()
lr = 0.001
device = 'cpu'
optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)
```

The ResNet-16 classifier reached an accuracy of 86.744% in test set, with a loss at `0.349`. The train/test loss curves and the outout of the last epoch are shown.  Obviously, the net has not reached its best performance, but due to time and computational resources limitation, I did not increase the `num_epochs`.

```
Epoch: 9
Train loss: 0.353 | Train Acc: 86.448% (2593449/3000000)
Test Loss: 0.349  | Test Acc: 86.744% (830639/957571)
```

![fcn1](asserts\resnet1.png)

![fcn1](asserts\resnet2.png)

### Regression

Here I use one linear layer in pytorch to act as the the regression model.

```
model = nn.Linear(87, 1)
```

Its hyperparameter is exactly the same as the ResNet classifier. However, it failed to obtain any feature by regressing directly from `pt`. Its loss curves are shown. The loss is calculated by `MSE loss`.

![img](data:image/png;base64,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%0AmJKHiIgU7P8BDt5oYZhVLcEAAAAASUVORK5CYII=)

## Task 2

I mainly reconstructed the feature in task 2. As I stated above, I reshape the feature into a 2D array to feed the ResNet16. And I move further here by constructing it into a `1*32*32` one channel image by reshaping and padding.

```
img_like_feature = np.pad(np.reshape(b, (10,10)), (11, 11), 'reflect') # from [87*1] to [32*32]
```

Here is an example of the `img_like_feature` I constructed:

![fcn1](asserts\feature.png)

Again, I use ResNet-16 to fit the dataset, with  `Adam` as the optimizer and   `CrossEntropyLoss` as the loss function. The full hyperparameters are:

```
batch_size = 256
num_epochs = 2
lossFunction = nn.CrossEntropyLoss()
lr = 0.001
device = 'cpu'
optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)
```

Pitfully, I do not have enough RAM to fine tune this model at a considerable epoch number, but it has the same trend as the ResNEt-16 in task 1.

```
Epoch: 0
Train loss: 0.917 | Train Acc: 56.830% (1704886/3000000)
Test Loss: 0.795  | Test Acc: 64.117% (613966/957571)

Epoch: 1
Train loss: 0.718 | Train Acc: 68.462% (2053859/3000000)
Test Loss: 0.670  | Test Acc: 70.886% (678782/957571)
```

## Task3

In this section, I build graph by hand-craft feature and KNN algorithm, and feed the graph dataset to the MPNN-based Quark/Gluon jet classifier.

### Hand-crafted Feature and KNN-based Graph Construction

The full jet dataset is giant and I cannot afford such an amount of RAM, so I only take the fitst file (100,000 jets) as my dataset. 

I notice that there are 14 type of `pdgid` in total and each jet no jets share the same number and type of `pdgid`. So, I calculate the type and number of `pdgid` of each jet (as a `14*1` array) and set them as a part of the feature of jet. Then I notice that the average pt is also a siginificant metric to tell the type of jet, and thus the average pt is taken into account. The hand-crafted feature is constructed by `[average_pt, pdgid_type_and_number_list]` (`15*1` array).

Then, I applied KNN algorithm to tell which K jets share the greatest similarity with each other.  The following `[100000*5]` list shows the neibours of each ject.

```
[[    0 17064 70444 11236 28423]
 [    1 53533 28137 21976 90660]
 [    2 99448 28005 32533 74298]
 ...
 [99997  3746  8665  4528 15614]
 [99998 15210 42989 42878 83043]
 [99999  1754 24424 64312 84631]]
```

### QG Jet Dataset

With the adjacent matrix, I build the QG jet dataset. The data structure of QG Jet Dataset is decribed as followed:

```
data = Data(x=feature.view(100000, -1), edge_index=adjacent_matrix.view(2,-1), y=label)
```

### MPNN

I implment the SEGAConv layer in my MPNN. The network structure is as followed.

```
MPNNet(
  (fc0): Linear(in_features=556, out_features=128, bias=True)
  (conv1): SAGEConv(
    (fc): Linear(in_features=128, out_features=128, bias=True)
    (act): ReLU()
    (update_fc): Linear(in_features=256, out_features=128, bias=False)
    (update_act): ReLU()
  )
  (conv2): SAGEConv(
    (fc): Linear(in_features=128, out_features=128, bias=True)
    (act): ReLU()
    (update_fc): Linear(in_features=256, out_features=128, bias=False)
    (update_act): ReLU()
  )
  (conv3): SAGEConv(
    (fc): Linear(in_features=128, out_features=128, bias=True)
    (act): ReLU()
    (update_fc): Linear(in_features=256, out_features=128, bias=False)
    (update_act): ReLU()
  )
  (fc1): Linear(in_features=256, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=1, bias=True)
  (act1): ReLU()
  (act2): ReLU()
)
```

And the hyperparameter is 

```
num_epochs = 10
batch_size = 256
lr = 0.01
device='cpu'
lossFunction = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
```

. However, the network failed to learn from the dataset, and it always predicts jets as `label:0`

## Optional

TBD