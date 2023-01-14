import torch
import torch.nn as nn
import torch.nn.functional as F
import os

FILE_PATH = os.path.split(__file__)[0]


class ConvBlockModel(nn.Module):

    def __init__(self, num_feat=3, pretrained=False, weight=None):
        super(ConvBlockModel, self).__init__()

        self.fc1 = nn.Linear(num_feat, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, 32)
        self.fc8 = nn.Linear(32, 1)

        self._initialize_weights()
        if pretrained:
            self._load_pretrained(weight)
            self.frozen_layer()

    # @torch.no_grad()
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu_(x)
        x = self.fc2(x)
        x = F.relu_(x)
        x = self.fc3(x)
        x = F.relu_(x)
        x = self.fc6(x)
        x = F.relu_(x)
        x = self.fc7(x)
        x = F.relu_(x)
        x = self.fc8(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def _load_pretrained(self, weight):
        print(weight)
        self.load_state_dict(torch.load(weight), strict=False)

    def frozen_layer(self):
        for p in self.parameters():
            p.requires_grad = False


def block_b(mobile_device, compute_device, scale=2, n_feat=4):
    return ConvBlockModel(num_feat=n_feat, pretrained=True,
                          weight=f'{FILE_PATH}/weights/{mobile_device}/{compute_device}/{compute_device}.pt')


if __name__ == "__main__":
    model = block_b(mobile_device='S21', compute_device='GPU', scale=2)
    test_input = torch.tensor([24, 144, 20, 24], dtype=torch.float)

