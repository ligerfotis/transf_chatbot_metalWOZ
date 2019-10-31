## Multi-Attention Transformer for Personal Chatbots on Multilingual Datasets
**Author: Lygerakis Fotios | Machine Learning Researcher**

**Research Associate at NCRS "Demokritos"**
 

### Description
This project utilize the multi-head attention transformer architecture to build a chatbot using big corpora of questions and responses.
 
* The transformer is based on this [tutorial](https://colab.research.google.com/github/tensorflow/examples/blob/master/community/en/transformer_chatbot.ipynb#scrollTo=WW3SeLDhAMJd).

* A lot of complementary functionalities are added regarding data preprossesing and the code is split into files to be more readable.

* We use a FastNet model to acquire pre-trained word and sub-word embeddings.

Languages:
* English
    * [MetalWOZ](https://www.microsoft.com/en-us/research/project/metalwoz/ ) dataset (~200K)
* Greek 
    * OpenSubtiltes Greek dataset subset (100K)
    
### Use
Python3
Install venv for python3 virtual environments: 

    sudo apt install -y python3-venv

Go to project directory: 

    cd transf_chatbot

Create a python virtual environment: 

    python3 -m venv project_env

Activate python: 

    source project_env/bin/activate

Install requirements: 

    pip3 install -r requirements.txt

Download the FastText models 

    cd embeddings
    
    wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
    
    gunzip cc.en.300.bin.gz
    
    wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.el.300.bin.gz
    
    gunzip cc.el.300.bin.gz

The code is configured to run on multiple GPU (if available).

Use the `config.yaml` file to configure the model, train and chat hyperparameters.

####Model

A model's hyperparameters are:

* NUM_LAYERS
* D_MODEL (depth of the model)
* NUM_HEADS
* UNITS
* DROPOUT
* lr:
    * initial_LR
    * decay_steps
    * decay_rate
    * staircase (True or False)
    * warmup_steps
    
**Important:** D_MODEL % NUM_HEADS == 0, because D_MODEL % NUM_HEADS = depth

Actually d_model is hard-coded from the dimensions of FastNet model which  is 300, but it would rise an error if not compatable.
    
####Train
You can train a model by simply running 

    python3 train.py

The hyperparameters for training in the configuration file are:

* MAX_LENGTH
* BATCH_SIZE
* BUFFER_SIZE
* EPOCHS
* resume_auto (True or False)
* resume_manual (True or False)
* manual_ckpt_filename
* manual_vocab_filename
* language (english or greek)

If resume_auto is True the program will automatically search for existing vocabulary file and checkpoint to load. If this fails it will create new ones.

After starting training, you can observe the epoch_loss, epoch_accuracy and learning rate curves by opening a new terminal and running from the project repository: 
    
    tensorboard --logdir logs/scalars/

The curves need some time to appear, because if starting from clean train there is no information to plot.

####Chat

You can chat with your chatbot after training your model or if you want to you can use a pretrained one.

You can simply run: 

    python3 chat.py


###Note:
The models need careful hyperparameter-tuning and long trainings (approximately 10000 epochs) depending on the size of the corpora. This can possibly take days on a GPU.