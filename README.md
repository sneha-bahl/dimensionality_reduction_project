# Project: Dimension Reduction
## Team Endgame: Jaskirat Singh (u7019589) and Quan Dao Minh (u7236134) and Sneha Bahl (u7006861)

## Instruction to run download dataset
1. Create a folder called **data**
2. You need to download 3 datasets: MNIST, FASHION MNIST, CIFAR-101.
- MNIST: you can download from the link: [mnist dataset](http://yann.lecun.com/exdb/mnist/). Please copy the download link in table below and download all following files:<br/>
    | Name | Link |
    | ------ | ------ |
    | `train-images-idx3-ubyte.gz` | yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz |
    | `train-labels-idx1-ubyte.gz` | yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz |
    | `t10k-images-idx3-ubyte.gz` | yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz |
    | `t10k-labels-idx1-ubyte.gz` | yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz |
Inside folder called **data** create folder  **mnist**. Unzip and store these files there

- FAHSION MNIST: you can download from the link: [fashion mnist dataset](https://github.com/zalandoresearch/fashion-mnist). Please copy the download link in table below and download all following files:<br/>
    | Name  | Link | 
    | --- |--- |
    | `train-images-idx3-ubyte.gz`  | fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz|
    | `train-labels-idx1-ubyte.gz`  | fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz|
    | `t10k-images-idx3-ubyte.gz`  | fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz|
    | `t10k-labels-idx1-ubyte.gz`  | fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz|


Inside folder called **data** create folder  **fashion-mnist**. Unzip and store these files there
- CIFAR-10: you can download from the link: [cifar dataset](https://www.cs.toronto.edu/~kriz/cifar.html). Please download the following files:<br/>
    | Name  | Link | 
    | --- |--- |
    | `CIFAR-10 python version`  | [Download](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)|
    

Inside folder called **data** create folder  **cifar10**. Unzip and store these files there

## Instruction to run the source code

Here is the table for each files in source code. You could look at the table and know the role of each file. Please note that the files are listed in the decreasing order of relevance, so please look at the topmost notebooks first to get an idea of implementation and results.
| File | Role |
| ------ | ------ |
| PCA.ipynb | Run PCA method |
| autoencoders_cifar.ipynb | Autoencoder results on CIFAR-10 |
| autoencoders_fashion_mnist.ipynb | Autoencoder results on Fashion-MNIST |
| autoencoders_mnist.ipynb | Autoencoder results on MNIST |
| quantitative_evaluation_cifar10.ipynb | All 3 quantitative evaluation criteria on CIFAR10 |
| quantitative_evaluation_fashion-mnist.ipynb | All 3 quantitative evaluation criteria on Fashion-MNIST |
| quantitative_evaluation_mnist.ipynb | All 3 quantitative evaluation criteria on MNIST |
| pca-autoencoder-equivalence_fashion-mnist.ipynb | Experiments for PCA-Autoencoder equivalence |
| autoencoders.py | Autoencoder definition class  |
| dune_ecology.ipynb | Dune Ecology analysis |
| evalutation.py | Reconstruction error plot function |
| method.py | dimensionality reduction modular function |
| pca_fashion-mnist.ipynb | PCA results on Fashion-MNIST  |
| pca_mnist.ipynb | PCA results on MNIST |
| quantitative_evaluation.ipynb | All 3 quantitative evaluation |
| t-SNE.ipynb | Run t-SNE method  |
| tsne_mnist.ipynb | cell |
| utils.py  | cell |
| visualize.py | cell |

 



