import torch

class BasicModel(torch.nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
        self.model_name = str(type(self))
        
    def load(self, path):
        self.load_state_dict(torch.load(path))
    
    def save(self, name=None):
        if name is None:
            prefix = './chechpoints' + self.model_name
        torch.save(self.state_dict(), prefix)
        return name