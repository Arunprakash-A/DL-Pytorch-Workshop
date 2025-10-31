## What is the use of `last_epoch` argument in the LRSchedulers like [ConstatnLR](https://docs.pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.ConstantLR.html)?
 1. First, the learning rate at the i-th epoch is a function of i (index)
 2. Therefore, it is important to keep track of the index-i (last index). Often it is last epoch
 3. It should be stored while checkpointing model parameters.

## I have a model containing multiple layers. I want to set a different learning rate for each layer. Is it possible to achieve that easily in PyTorch?
 1. Yes.
 2. Create parameter groups while creating the model. For example
    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim    
   
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            # Convolutional part
            self.conv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            # Fully connected part
            self.fc = nn.Sequential(
                nn.Linear(16 * 14 * 14, 10)
            )
    
        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
        
    # Initialize model
    model = SimpleCNN()
    
    # Create two parameter groups
    optimizer = optim.SGD([
        {'params': model.conv.parameters(), 'lr': 0.01},   # Group 1: conv layers
        {'params': model.fc.parameters(), 'lr': 0.1}       # Group 2: fc layers
    ], momentum=0.9)
    
    # Print parameter groups for clarity
    for i, group in enumerate(optimizer.param_groups):
        print(f"Parameter group {i+1}: learning rate = {group['lr']}")

    ```
 3. You can also pass a list of schedulers for each group while initializing the learning rate scheduler.
 4. Note: You can use only a learning rate scheduler (i.e., a function of epochs), not a function of validation loss (like [ReduceLROnPlateau](https://docs.pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html))
