import torch.nn as nn
import torch


class network(nn.Module):
    def __init__(self, numclass, feature_extractor):
        super(network, self).__init__()
        self.feature = feature_extractor
        self.fc = nn.Linear(512, numclass, bias=True)

    def forward(self, input):
        x = self.feature(input)
        x = self.fc(x)
        return x

    def Incremental_learning(self, old_class, numclass_classaug):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = old_class

        self.fc = nn.Linear(in_feature, numclass_classaug, bias=True)
        self.fc.weight.data[:out_feature] = weight[:out_feature]
        self.fc.bias.data[:out_feature] = bias[:out_feature]

    def saveOption(self, numclass):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features

        self.fc = nn.Linear(in_feature, numclass, bias=True)
        self.fc.weight.data = weight[:numclass]
        self.fc.bias.data = bias[:numclass]

    def feature_extractor(self,inputs):
        return self.feature(inputs)
