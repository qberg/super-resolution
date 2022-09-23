import torch.nn as nn

class SRCNN(nn.Module):

    def __init__(self,num_channels = 1):
        super(SRCNN, self).__init__()

        #Training -> (1,33,33)
        hidden_sizes = [64, 32]

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels,
                      hidden_sizes[0],
                      kernel_size=9,
                      padding=4,
                      stride=1), 
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_sizes[0],
                      hidden_sizes[1],
                      kernel_size=5,
                      padding=2,
                      stride=1),
            nn.ReLU()
        )
        self.conv3 = nn.Conv2d(hidden_sizes[1], 
                      num_channels, 
                      kernel_size=5, 
                      padding=2, 
                      stride=1)
        
    def forward(self, x):
        x   = self.conv1(x)
        x   = self.conv2(x)
        out = self.conv3(x)

        return out
