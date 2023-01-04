# Deep Learning Workshop using Pytorch 1.3 and above

  
 - You can access the slide deck here <a href=https://iitm-pod.slides.com/arunprakash_ai/pytorch> https://iitm-pod.slides.com/arunprakash_ai/pytorch </a>
 - Read the contents in the slide deck before using the following colab notebooks.
 
 ## Colab Notebooks
 1. **The Fuel: Tensors**
    - Understand the Pytorch architecture
    - Create Tensors of 0d,1d,2d,3d,... (a multidomensional array in numpy)
    - Understand the attributes : `storage, stride, offset, device`
    - Manipulate tensor dimensions
    - Operations on tensors
 2. **The Engine: Autograd**
    - A few more attributes of tensor : `requires_grad, grad, grad_fn, _saved_tensors, backward, retain_grad, zero_grad`
    - Computation gradph: Leaf node (parameters) vs non-leaf node (intermediate computation)
    - Accumulate gradient and update with context manager (torch.no_grad)
    - Implementating a neural network from scratch
    
    
 3. **The factory: nn.Module**
    - Brief tour into the source code of nn.Module 
    - Everything is a module (layer in other frameworks)
    - Stack modules by subclassing nn.Module and build any neural network
    - Managing data with `dataset` class and `DataLoader` class
    
 4. **Convolutional Neural Netowrk**
    - Using torchvision for datasets
    - build CNN and move it to GPU
    - Train and test
    - Transfer learning
    - Image segmentation

 
