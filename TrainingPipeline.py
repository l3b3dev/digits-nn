# Manages training life cycle
import os

import torch
from torch import nn
from torch.optim import SGD, Adam
from torchvision import datasets, transforms

from FeaturesController import FeaturesController
from GaussianNoiseTransform import GaussianNoiseTransform
from NN import DClassNet, DNet
from Plotter import Plotter


class TrainingPipeline:
    def __init__(self):
        self.models = [DClassNet(16 * 16, 65), DNet(16 * 16)]
        self.loss = [nn.NLLLoss(),  # negative log-likelihood loss, needed for classification of 65 objects
                     nn.MSELoss()]
        self.optimizer = self.init_optimizer(1)
        self.features_controller = FeaturesController()

    def init_optimizer(self, approach):
        self.optimizer = [SGD(self.models[approach - 1].parameters(), lr=0.001),
                          Adam(self.models[approach - 1].parameters(), lr=0.001)]

        return self.optimizer

    def plot_features(self, x):
        self.features_controller.plot_sift_descriptors(x)

    # Will try 2 approaches as listed in Step 02
    def get_model(self, approach):
        model = self.models[approach - 1]
        loss_fn = self.loss[approach - 1]
        optimizer = self.optimizer[approach - 1]
        return model, loss_fn, optimizer

    """ Negative log-likelihood loss and LogSoftmax together are the cross-entropy loss"""

    def train(self, x, y, model, optimizer, loss_fn, num_epochs=10000):

        loss_history = []
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            pred = model(x)
            loss_value = loss_fn(pred, y)
            loss_value.backward()
            optimizer.step()
            loss_history.append(loss_value)

        return loss_history

    @torch.no_grad()
    def val_loss(self, x, y, model, loss_fn):
        prediction = model(x)
        val_loss = loss_fn(prediction, y)
        return val_loss.item()

    # Create training and validation datasets and initialize data loaders
    def initialize_data(self, data_dir, sdev=0.):
        data_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor()
            ]),
            'test': transforms.Compose([
                transforms.ToTensor()
            ]),
            'raw': transforms.Compose([
                transforms.ToTensor()
            ])
        } if sdev == 0. else {
            'train': transforms.Compose([
                transforms.ToTensor()
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                GaussianNoiseTransform(std=sdev, k=25)
            ]),
            'raw': transforms.Compose([
                transforms.ToTensor(),
                GaussianNoiseTransform(std=sdev, k=25)
            ]),
        }

        # Create training and validation datasets
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                          ['train', 'test', 'raw']}
        # Create training and validation dataloaders
        image_dataloaders = {
            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=image_datasets[x].__len__(),
                                           shuffle=True if x == 'train' else False)
            for x
            in
            ['train', 'test', 'raw']}

        return image_datasets, image_dataloaders

    def load_all_data(self, image_dataloaders, kind='train'):
        # load
        X_train, Y_train = next(iter(image_dataloaders[kind]))
        # reshape and flatten
        X_train_f = torch.flatten(X_train[:, 0], start_dim=1)

        return X_train, Y_train, X_train_f

    @torch.no_grad()
    def predict(self, approach_number, model, x, x_test):
        if approach_number == 1:
            img = x.view(1, 16 * 16)
            y_pred = model(img)
            # Predictions are log-probabilities, do exponentials for real probabilities
            probs = list(torch.exp(y_pred).numpy()[0])
            # The index of the output pattern is found by locating the maximum value of y,
            # then finding the indx j of that value
            y = probs.index(max(probs))
            return x_test[y - 1]

        elif approach_number == 2:
            y_pred = model(x)
            return y_pred

    @torch.no_grad()
    def get_class_probabilities(self, classifier, x, x_test):
        classifier.eval()
        output = classifier(x)
        prediction = output.argmax(dim=1, keepdim=True)
        pred_images = x_test[prediction - 1]
        actuals = torch.round(x.view_as(pred_images))
        sm = torch.nn.Softmax()
        probabilities = sm(pred_images)

        return [p.item() for i in actuals for j in i for p in j], [p.item() for i in probabilities for j in i for p in
                                                                   j]

    @torch.no_grad()
    def get_image_probabilities(self, classifier, x, x_raw_f):
        classifier.eval()
        output = classifier(x)
        sm = torch.nn.Softmax()
        probabilities = sm(output)
        actuals = torch.round(x.view_as(x))

        return [j.item() for i in actuals for j in i], [j.item() for i in probabilities for j in i]

    def run_approach(self, approach_number, x_train_f, x_train, x_test, y_train, image_datasets):
        self.init_optimizer(approach_number)
        # setup labeling indexed list
        labels = [torch.LongTensor([float(image_datasets['train'].classes[lookup]) for lookup in y_train]),
                  x_train_f
                  ]
        model, loss_func, opt = self.get_model(approach_number)

        loss_history = self.train(x_train_f, labels[approach_number - 1],
                                  model, opt, loss_func, 8000)
        Plotter.plot_losses(loss_history)

        y_test_pred = self.predict(approach_number, model, x_train_f[0], x_test)
        y_pred = y_test_pred.reshape(16, 16)
        Plotter.plot_sample(x_train[0][0], y_pred)

        return model

    def load_pretrained(self, path):
        _models = []
        for approach_num in range(1, 3):
            model, _, _ = self.get_model(approach_num)
            model.load_state_dict(torch.load(f'{path}/model{approach_num}.pth'))
            model.eval()
            _models.append(model)

        return _models

    def render_test_data(self, m, x, x_raw):
        for i, model in enumerate(m):
            for x_test in x:
                # apply the model
                y_pred = self.predict(i + 1, model, x_test, x_raw)
                if i == 1:
                    Plotter.plot_sample(x_test.reshape(16, 16), y_pred.reshape(16, 16))

    @torch.no_grad()
    def get_fraction_statistics(self, x_test, y_pred):
        a = torch.round(x_test)
        b = torch.round(y_pred)
        diff = abs(a - b)

        blacks = a == 0
        whites = a == 1

        z = torch.logical_and(diff == 0, blacks).sum()
        fh = z.item() / blacks.sum().item()

        z = torch.logical_and(diff == 1, whites).sum()
        ffa = z.item() / whites.sum().item()

        return fh, ffa

    def compute_statistics(self, model, x, x_raw_f,  approach=1):
        Fh = []
        Ffa = []
        for x_test in x:
            # apply the model
            y_pred = self.predict(approach, model, x_test, x_raw_f)

            fh, ffa = self.get_fraction_statistics(x_test, y_pred)

            Fh.append(fh)
            Ffa.append(ffa)

        return Fh, Ffa

    def get_noise_stats(self, data_dir, model, sdevs, approach, x_raw_f, render=False):
        stats = {}
        img_probs = {}
        for sd in sdevs:
            image_datasets, loaders = self.initialize_data(data_dir, sdev=sd)
            X_test, Y_test, X_test_f = self.load_all_data(loaders, kind='test')

            # plot train data with labels
            if render:
                Plotter.plot_data(image_datasets, X_test, Y_test, kind='test')

            # calculate statistics
            stats[sd] = self.compute_statistics(model, X_test_f, x_raw_f, approach)
            img_probs[sd] = self.get_image_probabilities(model,
                                                         X_test_f,
                                                         x_raw_f) if approach == 2 else self.get_class_probabilities(
                model, X_test_f, x_raw_f)

        return stats, img_probs

    def save_models(self, path, models):
        for approach_num, model in enumerate(models):
            torch.save(model.state_dict(), f'{path}/model{approach_num + 1}.pth')
