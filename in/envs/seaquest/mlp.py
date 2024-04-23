import torch


class MLP(torch.nn.Module):
    def __init__(self, has_softmax=False, has_sigmoid=False, out_size=6, as_dict=False, logic=False, device=None):
        super().__init__()
        self.logic = logic
        self.as_dict = as_dict
        self.device = device
        encoding_base_features = 2
        encoding_entity_features = 2
        encoding_max_entities = 43
        self.num_in_features = (encoding_base_features + encoding_entity_features) * encoding_max_entities

        modules = [
            torch.nn.Linear(self.num_in_features, 40),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(40, 20),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(20, out_size)
        ]

        if has_softmax:
            modules.append(torch.nn.Softmax(dim=-1))
            
        if has_sigmoid:
            modules.append(torch.nn.Sigmoid())

        self.mlp = torch.nn.Sequential(*modules)

    def forward(self, state):
        features = state.float()
        y = self.mlp(features)
        return y
