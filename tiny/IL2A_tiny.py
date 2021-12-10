import torch
from torchvision import transforms
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim
from myNetwork_classaug import network
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import numpy as np
import os
import sys
from data_manager_tiny import *


class dualAug:
    def __init__(self, args, file_name, feature_extractor, task_size, device):
        self.args = args
        self.numclass = self.args.fg_nc
        self.task_size = task_size
        self.augnumclass = self.numclass + int(self.numclass*(self.numclass-1)/2)
        self.file_name = file_name
        self.model = network(self.augnumclass, feature_extractor)
        self.cov = None
        self.prototype = None
        self.class_label = None
        self.old_model = None
        self.device = device
        self.data_manager = DataManager()
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.train_loader = None
        self.test_loader = None

    def beforeTrain(self, current_task):
        self.model.eval()
        class_set = list(range(200))
        if current_task == 0:
            classes = class_set[:self.numclass]
        else:
            classes = class_set[self.numclass - self.task_size:self.numclass]
        print(classes)

        trainfolder = self.data_manager.get_dataset(self.train_transform, index=classes, train=True)
        testfolder = self.data_manager.get_dataset(self.test_transform, index=class_set[:self.numclass], train=False)

        self.train_loader = torch.utils.data.DataLoader(trainfolder, batch_size=self.args.batch_size,
                                                        shuffle=True, drop_last=True, num_workers=8)
        self.test_loader = torch.utils.data.DataLoader(testfolder, batch_size=self.args.batch_size,
                                                       shuffle=False, drop_last=False, num_workers=8)
        if current_task > 0:
            old_class = self.numclass - self.task_size
            self.augnumclass = self.numclass + int(self.task_size*(self.task_size-1)/2)
            self.model.Incremental_learning(old_class, self.augnumclass)
        self.model.train()
        self.model.to(self.device)

    def train(self, current_task, old_class=0):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=2e-4)
        scheduler = StepLR(opt, step_size=45, gamma=0.1)
        accuracy = 0
        for epoch in range(self.args.epochs):
            # print(100*'#')
            for step, data in enumerate(self.train_loader):
                images, target = data
                images, target = images.to(self.device), target.to(self.device)
                opt.zero_grad()
                loss = self._compute_loss(images, target, old_class)
                opt.zero_grad()
                loss.backward()
                opt.step()
            scheduler.step()
            if epoch % 10 == 0:
                accuracy = self._test(self.test_loader)
                print('epoch:%d,accuracy:%.5f' % (epoch, accuracy))
        self.protoSave(self.model, self.train_loader, current_task)
        return accuracy

    def _test(self, testloader):
        self.model.eval()
        correct, total = 0.0, 0.0
        for setp, data in enumerate(testloader):
            imgs, labels = data
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(imgs)
                outputs = outputs[:, :self.numclass]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        self.model.train()
        return accuracy

    def _compute_loss(self, imgs, target, old_class=0):
        imgs, target = imgs.to(self.device), target.to(self.device)
        imgs, target = self.classAug(imgs, target, mix_times=self.args.mix_times)
        output = self.model(imgs)
        loss_cls = nn.CrossEntropyLoss()(output/self.args.temp, target)
        if self.old_model == None:
            return loss_cls
        else:
            feature = self.model.feature(imgs)
            feature_old = self.old_model.feature(imgs)
            loss_kd = torch.dist(feature, feature_old, 2)

            proto_aug = self.prototype
            proto_aug_label = self.class_label
            proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().to(self.device)
            proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).to(self.device)
            index = list(range(old_class))
            np.random.shuffle(index)
            index = index[:self.args.batch_size]
            proto_aug = proto_aug.index_select(0, torch.tensor(index).to(self.device))

            proto_aug_label = proto_aug_label.index_select(0, torch.tensor(index).to(self.device))

            soft_feat_aug = self.model.fc(proto_aug)
            soft_feat_aug = soft_feat_aug[:, :self.numclass]

            ratio = 2.5
            isda_aug_proto_aug = self.semanAug(proto_aug, soft_feat_aug, proto_aug_label, ratio)
            loss_semanAug = nn.CrossEntropyLoss()(isda_aug_proto_aug/self.args.temp, proto_aug_label)
            return loss_cls + self.args.seman_weight*loss_semanAug + self.args.kd_weight*loss_kd

    def afterTrain(self):
        path = self.args.save_path + self.file_name + '/'
        if os.path.isdir(path) == False: os.makedirs(path)
        filename = path + '%d_model.pkl' % (self.numclass)
        self.model.saveOption(self.numclass)
        torch.save(self.model, filename)
        self.old_model = torch.load(filename)
        self.old_model.to(self.device)
        self.old_model.eval()
        self.numclass += self.task_size

    def protoSave(self, model, loader, current_task):
        features = []
        labels = []
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader):
                images, target = data
                feature = model.feature(images.to(self.device))
                if feature.shape[0] == self.args.batch_size:
                    labels.append(target.numpy())
                    features.append(feature.cpu().numpy())
        labels_set = np.unique(labels)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))

        prototype = []
        cov = []
        class_label = []
        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]
            prototype.append(np.mean(feature_classwise, axis=0))
            cov_class = np.cov(feature_classwise.T)
            cov.append(cov_class)
        if current_task == 0:
            self.cov = np.concatenate(cov, axis=0).reshape([-1, 512, 512])
            self.prototype = prototype
            self.class_label = class_label
        else:
            self.cov = np.concatenate((cov, self.cov), axis=0)
            self.prototype = np.concatenate((prototype, self.prototype), axis=0)
            self.class_label = np.concatenate((class_label, self.class_label), axis=0)

    def semanAug(self, features, y, labels, ratio):
        N = features.size(0)
        C = self.numclass
        A = features.size(1)
        weight_m = list(self.model.fc.parameters())[0]
        weight_m = weight_m[:self.numclass,:]
        NxW_ij = weight_m.expand(N, C, A)
        NxW_kj = torch.gather(NxW_ij, 1, labels.view(N, 1, 1).expand(N, C, A))
        CV = self.cov
        labels = labels.cpu()
        CV_temp = torch.from_numpy(CV[labels]).to(self.device)
        sigma2 = ratio * torch.bmm(torch.bmm(NxW_ij - NxW_kj, CV_temp.float()), (NxW_ij - NxW_kj).permute(0, 2, 1))
        sigma2 = sigma2.mul(torch.eye(C).to(self.device).expand(N, C, C)).sum(2).view(N, C)
        aug_result = y + 0.5 * sigma2
        return aug_result

    def classAug(self, x, y, alpha=20, mix_times=4):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        batch_size = x.size()[0]
        mix_data = []
        mix_target = []
        for _ in range(mix_times):
            index = torch.randperm(batch_size).to(self.device)
            for i in range(batch_size):
                if y[i] != y[index][i]:
                    new_label = self.generate_label(y[i].item(), y[index][i].item())
                    lam = np.random.beta(alpha, alpha)
                    if lam < 0.3 or lam > 0.7:
                        lam = 0.5
                    mix_data.append(lam * x[i] + (1 - lam) * x[index, :][i])
                    mix_target.append(new_label)

        new_target = torch.Tensor(mix_target)
        y = torch.cat((y, new_target.to(self.device).long()), 0)
        for item in mix_data:
            x = torch.cat((x, item.unsqueeze(0)), 0)
        return x, y

    def generate_label(self, y_a, y_b):
        if self.old_model == None:
            y_a, y_b = y_a, y_b
            assert y_a != y_b
            if y_a > y_b:
                tmp = y_a
                y_a = y_b
                y_b = tmp
            label_index = ((2 * self.numclass - y_a - 1) * y_a) / 2 + (y_b - y_a) - 1
        else:
            y_a = y_a - (self.numclass - self.task_size)
            y_b = y_b - (self.numclass - self.task_size)
            assert y_a != y_b
            if y_a > y_b:
                tmp = y_a
                y_a = y_b
                y_b = tmp
            label_index = int(((2 * self.task_size - y_a - 1) * y_a) / 2 + (y_b - y_a) - 1)
        return label_index + self.numclass


