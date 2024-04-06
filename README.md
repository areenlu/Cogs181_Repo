# Cogs181_Repo
 Final Project: Emotion Detection 

Requirements
Report format: Write a report with >1,500 words including main sections: a) abstract, b) introduction, c) method, d) experiment, e) conclusion, and f) references. You can follow the paper format as e.g leading machine learning journals such as Journal of Machine Learning Research (http://www.jmlr.org/) or IEEE Trans. on Pattern Analysis and Machine Intelligence (http://www.computer.org/web/tpami), or leading conferences like NeurIPS (https://papers.nips.cc/) and ICML (http://icml.cc/2016/?page_id=151). 

Submission: 

1. Submit your report to the Gradescope assignment Final Project.

2. (Mandatory) Attach your code as supplementary materials to the Gradescope assignment Final Project - Supplementary Materials . Alternatively, you can provide a link to the GitHub repository.

3. (Optional) You may also upload a supplemental presentation video in Final Project - Supplementary Materials. See below for more details.


Stanford CS231n project design tips.

Stanford CS231n project descriptions that are helpful.

Bonus points: If you feel that your work deserves bonus points due to reasons such as: a) novel ideas and applications, b) large efforts in your own data collection/preparation, c) state-of-the-art results on your applications, or d) new algorithms or neural network architecture, please create a "Bonus Points" section to specifically describe why you deserve bonus points. In general, we evaluate your justifications based on the review guidelines based on e.g. CVPR/NeurIPS/ICCV/ICLR.

Templates  (using Google Docs or Word is fine too):

NeurIPS: https://neurips.cc/Conferences/2023/PaperInformation/StyleFiles

ICML: https://media.icml.cc/Conferences/ICML2023/Styles/icml2023.zip

ICLR: https://github.com/ICLR/Master-Template/raw/master/iclr2024.zip

In addition, there will be an optional presentation for the final project to receive bonus points. You can either submit a 3-5 minute short video clip to Gradescope as supplementary materials to your report or to have a physical presence for your presentation.

Note that the requirement for the word count (>1,500)  only applies to a single-student project. For team-based projects, each team only needs to write one final report but the role of each team member needs to be clearly defined and specified. The final project report is also supposed to be much longer than 1,500 words, depending upon how many (maximum 2) members there are in your team.

Word count:

One-person team: >1,500

Two-persons team: > 2,200

See below a link about writing a scientific paper: http://abacus.bates.edu/~ganderso/biology/resources/writing/HTWtoc.html The format of your references can be of any kind that is adopted in the above journals or conferences.

Grading: The merit and grading of your project can be judged from aspects described below that are common when reviewing a paper: 

1. Interestingness of the problem you are studying. (10 points).

2. How challenging and large is the dataset you are studying? (10 points)

3. Any aspects that are new in terms of algorithm development, uniqueness of the data, or new applications? (20 points)

Note that we encourage you to think of something beyond just downloading existing code and training on standard benchmarks. Basically, you are expected to complete a report that is worth reading (even not publishable to some extent). If you have done a good job of relatively thorough investigation of e.g. different architectures and different parameters, it is considered to be somewhat "new". The experiences you have are worth reading for others who have not tried them before. When someone is reading your report, he/she will feel like something worthy is there including your own attempts for algorithms, a non-standard dataset or application, a general conclusion about your parameter tuning, what the neural network structure might be a better choice, etc.

In a nutshell, this definition of "new" is somewhat different from the aspect of "being novel" when reviewing a paper that is submitted to e.g. NeurIPS. Will add this part to the project description.

4. Is your experimental design comprehensive? Have you done thorough experiments in tuning hyper-parameters? (30 points)

Tuning hyper-parameters in your final project will need to be more comprehensive than what was done in the homework.

For example, if you are performing CNN classification on the Tiny ImageNet dataset, some options to consider include

a. Comparing two different architectures chosen from e.g. LeNet, AlexNet, VGG, GoogleNet, or ResNet

b. Trying to vary the number of layers.

c. Trying to adopt different optimization methods, for example, Adam vs. stochastic gradient descent

d. Trying different pooling functions, average pooling, max pooling, stochastic pooling

e. Trying to use different activation functions such as ReLu, Sigmoid, etc.

 

See e.g. how the significance was justified in the ResNet paper (you don't have to follow this paper though): https://arxiv.org/pdf/1512.03385.pdf

 

5. Is your report written in a professional way with sections including abstract, introduction, method/architecture description, experiments ( data and problem description, hyper-parameters, training process, etc.), conclusion, and references? (30 points)

6. Bonus points will be assigned to projects that have adopted new methods, worked on novel applications, and/or have done a thorough comparison against the existing methods and possible choices.

There will be three options for the final project (if you have your own idea, please come to talk to me):

Option (1): Convolutional neural networks Train a convolutional neural networks method on Cifar-10 or Tiny ImageNet dataset (http://pages.ucsd.edu/~ztu/courses/tiny-imagenet-200.zip) You can choose any deep learning platforms including PyTorch (https://pytorch.org), TensorFlow (https://www.tensorflow.org), MxNet (https://github.com/dmlc/mxnet), Theano (http://deeplearning.net/software/theano/), Caffe (http://caffe.berkeleyvision.org/), and MatConvNet (http://www.vlfeat.org/matconvnet/). But using PyTorch is highly recommended.

See in the link other possible platforms you can use: http://deeplearning.net/software_links/ You can train a model by building your own network structure or by adopting/following standard networks like AlexNet, GoogLeNet, VGG, etc. You are also welcome to use datasets other than ImageNet, e.g. CIFAR-10, and Kaggle datasets (https://www.kaggle.com/) such as the deep sea image competition: https://www.kaggle.com/c/datasciencebowl

Modifications you can make: different activation functions, initialization methods, optimizers, pooling, number of layers, dropout, normalization (batch, layer, group) etc.

Interesting directions to pursue:

Networks visualization: VisualizingAndUnderstanding, Top-down attention

Compact networks: MobileNets

Binary networks: Binarized network

PyTorch for Cifar: https://github.com/kuangliu/pytorch-cifar

Some github resources about the Tiny ImageNet classification for PyTorch:

https://github.com/tjmoon0104/pytorch-tiny-imagenet

Option (2): (Individual only, no teamwork) Char RNN. You can read more about “The Unreasonable Effectiveness of Recurrent Neural Networks” at the link http://karpathy.github.io/2015/05/21/rnn-effectiveness/ 

Code is available at https://github.com/karpathy/char-rnn or https://github.com/jcjohnson/torch-rnn or https://github.com/sherjilozair/char-rnn-tensorflow (You may also use any other char-rnn implementation). Tiny Shakespeare dataset is available at https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt and complete Sherlock Holmes is available at https://sherlock-holm.es/stories/plain-text/cnus.txt You can also try out other interesting applications as described in the post, such as Wikipedia, Algebraic Geometry (Latex) or Linux Source Code, but you need to collect the dataset by yourself. You need to work on at least one dataset/application and try to produce meaningful results by using the char rnn model.

Option (3): (please discuss it with the instructor when planning). A topic of your own about visual, acoustic, language, and other data modeling using modern deep learning techniques including convolutional neural networks, Transformers, recurrent neural networks, auto-encoders, etc. You can look for interesting topics on recent NeurIPS, CVPR, ICLR, ICCV, ACL, AAAI, etc.
