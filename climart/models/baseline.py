import torch
import torch.nn as nn
import torch.nn.functional as F


class LW_abs(nn.Module):
    """ 
    Inputs: 18 (Temp, Pres, and Gasesous concentrations) as inputs
    Outputs: 256 (g-points LW absorption cross section)

    """
    def __init__(self, input_size):
        super(LW_abs, self).__init__()
        
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, 58)  #Not sure about the input shape, depends on nlevels
        self.fc2 = nn.Linear(58, 58)
        self.fc3 = nn.Linear(58, 256)


    def forward(self, x):

        x = F.softsign(self.fc1(x))
        x = F.softsign(self.fc2(x))
        x = self.fc3(x)
        return x


class LW_ems(nn.Module):
    """
    Inputs: 18 (Temp, Pres, and Gasesous concentrations) as inputs
    Outputs: 256 (g-points LW emission)

    """
    def __init__(self, input_size):
        super(LW_ems, self).__init__()
        
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 256)


    def forward(self, x):

        x = F.softsign(self.fc1(x))
        x = F.softsign(self.fc2(x))
        x = self.fc3(x)
        return x


class SW_abs(nn.Module):
    """
    Inputs: 7 (Temp, Pres, and Gasesous concentrations) as inputs, SW takes lesser gases than LW
    Outputs: 224 (g-points SW absorption cross section)

    """
    def __init__(self, input_size):
        super(SW_abs, self).__init__()
        
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, 48)        
        self.fc2 = nn.Linear(48, 48)
        self.fc3 = nn.Linear(48, 224)


    def forward(self, x):

        x = F.softsign(self.fc1(x))
        x = F.softsign(self.fc2(x))
        x = self.fc3(x)
        return x



class SW_rcs(nn.Module):
    """
    Inputs: 7 (Temp, Pres, and Gasesous concentrations) as inputs, SW takes lesser gases than LW
    Outputs: 224 (g-points SW rayleigh cross section)

    """
    def __init__(self, input_size):
        super(SW_rcs, self).__init__()
        
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 224)


    def forward(self, x):

        x = F.softsign(self.fc1(x))
        x = F.softsign(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    net = SW_rcs(180)
    print(net)
   

    input =  torch.randn(180)
    out = net(input)
    print(out)
#    params = list(net.parameters())
#    print(params)
