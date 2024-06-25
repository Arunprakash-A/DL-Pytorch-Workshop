# Deep Learning Workshop using Pytorch 1.3 and above

* UPDATE: Added Hugging Face Notebooks. HF is an additional layer on the Pytorch (and Tensorflow). It does a lot of Heavy lifting and makes our job of building prototypes (trai-test-infer-host) a lot easier!

  
 - You can access the slide deck that covers Pytorch here <a href=https://iitm-pod.slides.com/arunprakash_ai/pytorch> https://iitm-pod.slides.com/arunprakash_ai/pytorch </a>
 - You can access the slide deck that covers various concepts related to Transformers [Here](https://iitm-pod.slides.com/arunprakash_ai/transformers-distilled-hf-workshop-iitmbs) </a>
 - You should read the contents in the slide decks before using the following colab notebooks.
 - Once you get a good grip on the first four modules, you can easily walk through the documentation or other code to build an application. I will keep updating this repository.
 - [Recorded videos](https://drive.google.com/drive/folders/1o6AS8QE0xHpLS99kMlnDfzQp3VIrsQ1p?usp=sharing) 
 ## Colab Notebooks
 1. **The Fuel:** [Tensors](https://colab.research.google.com/drive/179Gv23AcUDCOhHt82msbstQZrbzS6Qn4?usp=sharing)
    - Understand the Pytorch architecture
    - Create Tensors of 0d,1d,2d,3d,... (a multidimensional array in numpy)
    - Understand the attributes: `storage, stride, offset, device`
    - Manipulate tensor dimensions
    - Operations on tensors
 2. **The Engine:** [Autograd](https://colab.research.google.com/drive/12h5SZ0FaZXUYzEP5DM2GTIg2KIeFfiG4?usp=sharing)
    - A few more attributes of tensor : `requires_grad, grad, grad_fn, _saved_tensors, backward, retain_grad, zero_grad`
    - Computation graph: Leaf node (parameters) vs non-leaf node (intermediate computation)
    - Accumulate gradient and update with context manager (torch.no_grad)
    - Implementing a neural network from scratch
    
    
 3. **The factory:** [nn.Module](https://colab.research.google.com/drive/1bz87qDYbidxskT6pkxJ-pRaF39qFteMv?usp=sharing), [Data Utils](https://colab.research.google.com/drive/1A9D0wzQ93Bl06cpAYhFvYO2cGe8sasof?usp=sharing)
    - Brief tour into the source code of nn.Module 
    - Everything is a module (layer in other frameworks)
    - Stack modules by subclassing nn.Module and build any neural network   
    - Managing data with `dataset` class and `DataLoader` class
    
 4. **Convolutional Neural Network** [Image Classification](https://colab.research.google.com/drive/1M9ha7mZ-42UKUFZGee5QeKHbdNoo3U51?usp=sharing)
    - Using torchvision for datasets
    - build CNN and move it to GPU
    - Train and test
    - Transfer learning
    - Image segmentation


5. **Recurrent Neural Network** [Sequence classification](https://colab.research.google.com/drive/1OAraEdQfr_rhXGeANZ83v5gJ4Kt14aAr?usp=sharing)
    - torchdata
    - torchtext
    - Embedding for words
    - Build RNN
    - Train,test, infer
  
6. **Using pre-trained models from Hugging Face**[Two Core Modules of HF](https://drive.google.com/file/d/1dAPaHzqLrRWsF4lAq9ydtKAK5F81zlIm/view?usp=sharing)
     - Tokenizer (AutoTokenizer)
     - Transformers (AutoModel)
 
