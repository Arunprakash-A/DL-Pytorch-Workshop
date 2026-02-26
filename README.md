# Deep Learning with Pytorch and Hugging Face

 - You can access the slide deck that covers Pytorch [Here](https://iitm-pod.slides.com/arunprakash_ai/pytorch)
 - You can access the slide deck that covers various concepts related to Transformers [Here](https://iitm-pod.slides.com/arunprakash_ai/transformers-distilled-hf-workshop-iitmbs)
 - It is recommended to read the slide decks before using the following colab notebooks 
 - Once you get a good grip on the first four modules, you can easily walk through the documentation or other code to build an application. I will keep updating this repository.
 - [Recorded videos](https://drive.google.com/drive/folders/1o6AS8QE0xHpLS99kMlnDfzQp3VIrsQ1p?usp=sharing)
 - Some recommendations 
     - Always start with any one of the following optimizers (priority in order): AdamW/Adam/SGD.
     - PyTorch implements the optimizers in three ways: Forloop (slow), foreach (fast) and fused (faster)
     - AdamW/Adam/SGD are implemented using a fused kernel and are also stable (extremely important) - [Doc](https://docs.pytorch.org/docs/stable/optim.html#:~:text=We%20have%203,top%20of%20that.)
       
 ## Colab Notebooks
 1. **The Fuel:** [Tensors](https://colab.research.google.com/drive/179Gv23AcUDCOhHt82msbstQZrbzS6Qn4?usp=sharing)
    - **Difficulty Level:** Easy if you have prior experience using Numpy or TensorFlow
    - Understand the Pytorch architecture
    - Create Tensors of 0d,1d,2d,3d,... (a multidimensional array in numpy)
    - Understand the attributes: `storage, stride, offset, device`
    - Manipulate tensor dimensions
    - Operations on tensors
 2. **The Engine:** [Autograd](https://colab.research.google.com/drive/12h5SZ0FaZXUYzEP5DM2GTIg2KIeFfiG4?usp=sharing)
    -  **Difficulty Level:** Hard, requires a good understanding of backprop algorithm. However, you can skip this and still follow the subsequent notebooks easily.
    - A few more attributes of tensor : `requires_grad, grad, grad_fn, _saved_tensors, backward, retain_grad, zero_grad`
    - Computation graph: Leaf node (parameters) vs non-leaf node (intermediate computation)
    - Accumulate gradient and update with context manager (torch.no_grad)
    - Implementing a neural network from scratch
    
    
 3. **The factory:** [nn.Module](https://colab.research.google.com/drive/1bz87qDYbidxskT6pkxJ-pRaF39qFteMv?usp=sharing), [Data Utils](https://colab.research.google.com/drive/1A9D0wzQ93Bl06cpAYhFvYO2cGe8sasof?usp=sharing)
    - **Difficulty Level:** Medium
    - Brief tour into the source code of nn.Module 
    - Everything is a module (layer in other frameworks)
    - Stack modules by subclassing nn.Module and build any neural network   
    - Managing data with `dataset` class and `DataLoader` class
    
 4. **Convolutional Neural Network** [Image Classification](https://colab.research.google.com/drive/1M9ha7mZ-42UKUFZGee5QeKHbdNoo3U51?usp=sharing)
    - **Difficulty Level:** Medium
    - Using torchvision for datasets
    - build CNN and move it to GPU
    - Train and test
    - Transfer learning
    - Image segmentation
---
Update
 - You can use various learning rate schedulers such as ExponentialLR, CosineAnnealing and so on. You just need to call `scheduler.step()` after `optimizer.step`. Refer to the documentation [here](https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
- A slight change in instantiating pre-trained models [Refer](https://www.learnpytorch.io/06_pytorch_transfer_learning/)
---
5. **Recurrent Neural Network** [Sequence classification](https://colab.research.google.com/drive/1OAraEdQfr_rhXGeANZ83v5gJ4Kt14aAr?usp=sharing)
    - **Difficulty Level:** Hard for pre-processing part, Medium for model building part
    - torchdata
    - torchtext
    - Embedding for words
    - Build RNN
    - Train,test, infer
---
Please take a look at the [official tutorial series](https://pytorch.org/tutorials/distributed/home.html) if you want to perform distributed training using a multi-GPU or multi-node setup in PyTorch (requires minimal modifications to the existing code). It covers various approaches, including:
 - Distributed Data-Parallel (DDP) [single-node/multi-node](https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/README.md)
 - Fully Sharded Data Parallel (FSDP)
 - Model, Tenosr and Pipeline parallelism
 - [TorchAO](https://docs.pytorch.org/ao/stable/index.html)  for low-precision training, serving and optimization <br> <br>
Now, let's move on to the Hugging Face library, which further simplifies these training strategies
---
6. **Using pre-trained models** [Notebook](https://drive.google.com/file/d/1dAPaHzqLrRWsF4lAq9ydtKAK5F81zlIm/view?usp=sharing)     
     - **Difficulty Level:** Easy
     - AutoTokenizer
     - AutoModel
7. **Fine-Tuning Pre-Trained Models** [Notebook](https://colab.research.google.com/drive/1ccfdwR6Olvgh2-sm8BeqQeUdp-itKoYX?usp=sharing)
     - **Difficulty Level:** Medium
     -  datasets
     -  tokenizer
     -  data collator with padding
     -  Trainer
8. **Loading Datasets** [Notebook](https://colab.research.google.com/drive/16U91dlO9CawJUCdqzSbaKjDgusuBGcJF?usp=sharing)
     - **Difficulty Level:** Easy
     - Dataset from local data files
     - Dataset from Hub
     - Preprocessing the dataset: Slice, Select, map, filter, flatten, interleave, concatenate
     - Loading from external links
9. **Build a Custom Tokenizer for translation task** [Notebook](https://colab.research.google.com/drive/1YizCe9z6GzCkoWkfSp9HHFo51T5hY-uh?usp=sharing)
     - **Difficulty Level:** Medium
     - Translation dataset as running example
     - Building the tokenizer by encapsulating the Normalizer, pre-tokenizer and tokenization algorithm (BPE)
     - Locally Save and Load the tokenizer
     - Using it in the Transformer module
     - Exercise: Build a Tokenizer with shared vocabulary.
10. **Training Custom Seq2Seq model  using Vanilla Transformer Architecture** [Notebook](https://colab.research.google.com/drive/1MbUDJup6i1MgXimMfo1mLeWpsH7nK-_K?usp=sharing)
     - **Difficulty Level:** Medium, if you know how to build models in PyTorch.
     - Build Vanilla Transformer architecture in Pytorch
     - Create a configuration file for a model using __PretrainedConfig__ class
     - Wrap it by HF __PreTrainedModel__ class
     - Use the custom tokenizer built in the previous notebook
     - Use Trainer API to train the model
11. **Gradient Accumulation - Continual Pre-training** [Notebook](https://gist.github.com/Arunprakash-A/c27ebe06e6c8fbd21263fc54013bbf49)
     - **Difficulty Level:** Easy
     - Understand the memory requirement for training and inference
     - Understand how gradient accumulation overcomes the limited memory      

---
CUDA Resources
---
Pytorch updated [Cuda Semantics](https://docs.pytorch.org/docs/main/notes/cuda.html) page on Aug 07 2025. If you are using Multiple GPUs, you must read it before starting to write code. Don't assume!
