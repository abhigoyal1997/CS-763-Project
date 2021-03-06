# Visual Question Answering

## Objective
The goal of Visual Question Answering is to be able to understand the semantics of scenes well enough to be able to answer open-ended, free-form natural language questions (asked by humans) about images. We aim to build and evaluate deep neural networks and compare them with certain baselines and state of the art models for the task. 

## Solution Approaches

### VGG + GLoVe embedding + MLP
We employ fine-tuning of pretrained VGG model to extract image features. GloVe Text embeddings are used to map each word of the question to a 300-dimensional vector. The question text embedding sequence is passed through a LSTM to get a thought vector representing question information in some lower dimension. 
The feature vectors from the above two subnetworks are concatenated and then passed through a MLP to get a 1-class classification (Probability of YES). 

### Self-trained ConvNet + GloVe embedding + MLP
We replace the pretrained model of VGGNet above by our own ConvNet with 3 convolution layers and appropriate batchnorm, dropout layers in between. The rest is same as the model above. This would hopefully accomplish learning of image features specific to our task. 

### Self-trained ConvNet + Self-trained embeddings + MLP
Similar to the idea of replacing VGGNet with our own ConvNet, we tried replcing GloVe embedding with an embedding network to learn text embeddings. It is stated in some papers that pretrained embedings tend to map words like Yes, No to similar embeddings which should be clearly differentiated in our application.

### VGG + GLoVe embedding + Image as initial memory of LSTM
Idea for this model was inspired from Exploring Models and Data for Image Question Answering(https://arxiv.org/pdf/1505.02074.pdf). Instead of taking image and text feature map separately and then passing it through an MLP, we first take the feature embedding of image. We initialise the hidden state of the text LSTM with this feature map and pass the question words as input to this LSTM. The idea is that LSTM will build on the image information the information from the text embeddings and thus retain relevant information from both entities. 

### VGG + Self-trained embedding + SAN
Idea taken from Stacked Attention Networks for Image Question Answering(https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Yang_Stacked_Attention_Networks_CVPR_2016_paper.pdf). Attention networks have been shown to work efficiently on tasks involving information extraction on images, like image captioning. So here the question + the image are used to get attention distribution on the image, thus focusing on only those parts of the image which are relevant to the question. 



## Dataset
We have used the datasets put up by the VQA group, Virginia Tech university for training our model and also testing. The datasets can be found at  https://visualqa.org/download.html
We use Balanced Binary Abstract Scenes from the dataset which has only binary questions like "Are there flowers at the foot of the tree?"

The dataset consists of 20,629 Training images , 10,696 Validation images, 22,055 Training questions, 11,328 Validation questions.
Each question has 10 answers, taken from 10 independent human subjects. Thus there are
 220,550 Training annotations, 113,280 Validation annotations. The distribution is 121556 Yes answers, 128894 No answers


## Requirements 
Kindly use the requirements.txt to set up your machine for replicating this 
experiment. some dependendecies are :

```
Package              Version    
-------------------- -----------      
jsonschema           2.6.0           
matplotlib           3.0.3        
numpy                1.16.2     
pandas               0.24.2     
pandocfilters        1.4.2          
Pillow               5.4.1           
scipy                1.2.1         
spacy                2.1.3        
tensorboard          1.13.1     
tensorboardX         1.6        
tensorflow           1.13.1     
tensorflow-estimator 1.13.0                  
torch                1.0.1.post2
torchvision          0.2.2.post3 
tqdm                 4.31.1

```


## Instructions

```
usage: main.py [-h] [-p DEVICE] {train,test} ...

positional arguments:
  {train,test}

optional arguments:
  -h, --help            show this help message and exit
  -p DEVICE, --device DEVICE

```

Training command : 
```
usage: main.py train [-h] [-mc MODEL_CONFIG] [-c COMMENT] [-d DATA_ROOT]
                     [-ds DS] [-s TRAIN_SPECS]
                     model_path

positional arguments:
  model_path

optional arguments:
  -h, --help            show this help message and exit
  -mc MODEL_CONFIG, --model-config MODEL_CONFIG
  -c COMMENT, --comment COMMENT
  -d DATA_ROOT, --data-root DATA_ROOT
  -ds DS, --train-size DS
  -s TRAIN_SPECS, --train-specs TRAIN_SPECS

```

Testing command : 
```
usage: main.py test [-h] [-d DATA_ROOT] [-i] model_path

positional arguments:
  model_path

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_ROOT, --data-root DATA_ROOT
  -i, --test-init

```

## Results and Observations

The accuracy of any trained model on a test image and question is calculated as :
![plot](https://github.com/abhigoyal1997/CS-763-Project/blob/master/images/accuracy.png)

The following image shows the plots for training loss, accuracy and validation loss, accuracy for Self-trained ConvNet + GLoVe embedding + MLP. One can observe that the validation accuracy for this model reaches  around 63%, which is about the best that this model can achieve (https://imatge.upc.edu/web/sites/default/files/pub/xRoldan.pdf).
![plot](https://github.com/abhigoyal1997/CS-763-Project/blob/master/results/training.png)

This image shows plots for training loss, accuracy and validation loss, accuracy for SAN. Three different learning rates have been combined in the plot i.e 1e-3, 1e-4, 1e-5. 
![plot](https://github.com/abhigoyal1997/CS-763-Project/blob/master/results/training_san.png)

## Discussions
The naive model that we tried (Self-trained ConvNet + GloVe embedding + MLP) achieved a decent accuracy compared to its capacity. However the validation loss curve looks a bit discouraging since the val_loss keeps on increasing from the beginning. This problem was even more amplified when we tried Stacked Attention Network (SAN).

We found that the existing papers (references provided) are a bit incomplete. Most papers only discuss about the architecture, but don't mention the details about their training parameters. Some don't even mention the loss that they've used. [Some papers tend to contradict each other in what might and might not help the training of the networks]

We tried many things to alleviate this problem of 'overfitting from the beginning'. We experimented extensively with Dropout and Batchnorm, but couldn't get major performance gains. For ensuring sanity, we did the 'overfitting experiment', and it went as planned. The network was quickly able to overfit on a small part of the test data. We tried different flavours of cross-entropy loss as well, but even that didn't go very well. 
From this, our conclusion says that the overall performance of the model is a little too sensitive to the parameters, and this might have hindered us from reproducing the results that the SAN paper mentions. In fact, VQA as a task (with the current approaches) might be too sensitive to the hyperparameters, and the paper "Show, Ask, Attend, Tell" says that for VQA, the devil is in the details.

One major dubiousness is the loss that the references have used. The paper on Stacked Attention Networks doesn't mention the word loss at all. We studied "Show, Ask, Attend, Tell" which mentions the following loss :
![plot](https://github.com/abhigoyal1997/CS-763-Project/blob/master/images/loss1.png)

We believe there might be a problematic artifact with this. Suppose if for an image-question pair, 8 of the 10 ground truth labels say 'Yes' and 2 of the 10 say 'No', then the issue will be that the probability of 'Yes' predicted by the model won't go above a threshold. Since for high p_yes values, 8/10(-log p) will be low but 2/10(-log(1-p)) will be too high, and this will send a gradient signal back to decrease the value of p. This might serve the purpose of avoiding too confident predictions, but the problem is that the confidence doesn't rise to that too high range from the beginning! We implemented this loss but it didn't work, as shown in the training curves of SAN.

We also tried to modify it to :
![plot](https://github.com/abhigoyal1997/CS-763-Project/blob/master/images/loss2.png)
But that also didn't perform very well.

## References

We take ideas from the following research publications for our VQA task

* https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Antol_VQA_Visual_Question_ICCV_2015_paper.pdf VQA: Visual Question Answering (2015)
* https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Malinowski_Ask_Your_Neurons_ICCV_2015_paper.pdf Ask Your Neurons: A Neural-based Approach to Answering Questions about Images (2015)
* https://arxiv.org/pdf/1505.02074.pdf Exploring Models and Data for Image Question Answering (2015)
* https://arxiv.org/pdf/1601.01705.pdf Learning to Compose Neural Networks for Question Answering (2016)
* http://proceedings.mlr.press/v48/xiong16.pdf Dynamic Memory Networks for Visual and Textual Question Answering (2016)
* https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Yang_Stacked_Attention_Networks_CVPR_2016_paper.pdf Stacked Attention Networks for Image Question Answering (2016)
* https://arxiv.org/pdf/1704.03162.pdf Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering (2017)
