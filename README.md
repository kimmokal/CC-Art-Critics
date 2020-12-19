# CC-Art-Critics
Group project for the Computational Creativity course at the University of Helsinki (autumn 2020). "The Art Critics" group consists of Juuso Lassila and Kimmo Kallonen.

We present a computationally creative system for generating abstract artworks with the help of a [pre-trained BigGAN](https://github.com/huggingface/pytorch-pretrained-BigGAN). The architecture of the system is shown below.

![Architecture](CC_architecture.png)

The system is composed of four connected modules:
1. Input transformer
2. Generator (BigGAN)
3. Abstract art discriminator
4. Feedback predictor

The input space of the BigGAN is explored with the help of the input transformer. The goal of the input transformer is to simultaneously maximize the abstractness of the generated images, as evaluated by the abstract art discriminator, and the predicted feedback of an "art critic", as evaluated by the feedback predictor.

## Shortcut: Quick image generation

There are pre-trained models available ... 

## Full procedure:

#### Step 1 - Initialization

First clone the repository and install the required packages.
```
git clone https://github.com/kimmokal/CC-Art-Critics
```

It is advisable to create a Python 3 virtual environment in which to install the packages.
```
pip install -r requirements.txt
```

### Step 2 - Download the training data set

...

### Step 3 - Give ratings to the abstract images

...

### Step 4 - Train the abstract art discriminator

...

### Step 5 - Train the feedback predictor

...

### Step 6 - Train the input transformer

...

### Step 7 - Generate images

...
