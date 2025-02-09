from itertools import cycle

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


class Plotter:
    @staticmethod
    def plot_data(image_datasets, x_train, y_train, title, kind='train'):
        # plot train data with labels
        R, C = 1, x_train.size(0)
        fig, ax = plt.subplots(R, C, figsize=(20, 10))
        fig.suptitle(title)
        for i, plot_cell in enumerate(ax):
            plot_cell.grid(False)
            plot_cell.axis('off')
            plot_cell.set_title(image_datasets[kind].classes[y_train[i].item()])
            plot_cell.imshow(x_train[i][0], cmap='gray')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_losses(loss_history):
        plt.plot(loss_history)
        plt.title('Loss variation over increasing epochs')
        plt.xlabel('epochs')
        plt.ylabel('loss value')
        plt.show()

    @staticmethod
    def plot_stats(fh, ffa, approach):
        network = "Heteroassociative Multi-Layer Neural Network" if approach == 1 else "Autoassociative Multi-Layer Neural Network"
        plt.scatter(ffa, fh, facecolors='none', edgecolors='r')
        plt.title(f'Fh vs Ffa for Approach {network}')
        plt.xlabel('Ffa')
        plt.ylabel('Fh')
        plt.show()

    @staticmethod
    def plot_sample(x, y):
        # # plot predicted data with
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Predicted Data')
        ax1.grid(False)
        ax1.axis('off')
        ax1.set_title('Actual')
        ax1.imshow(x, cmap='gray')

        ax2.grid(False)
        ax2.axis('off')
        ax2.set_title('Predicted')
        ax2.imshow(y.detach().numpy(), cmap='gray')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_class_roc(actuals, class_probabilities):
        fpr, tpr, _ = roc_curve(actuals, class_probabilities)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

    @staticmethod
    def plot_noise_roc(probs, approach):
        network = "Heteroassociative Multi-Layer Neural Network" if approach == 1 else "Autoassociative Multi-Layer Neural Network"
        plt.figure(figsize=(8, 8))
        lw = 1

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'yellow', 'blue', 'red', 'green','pink'])
        for (noise, (actuals, class_probabilities)), color in zip(probs.items(), colors):
            fpr, tpr, _ = roc_curve(actuals, class_probabilities)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color,
                     lw=lw, label='ROC curve for noise level {0} (area = {1:0.2f})'.format(noise, roc_auc))

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Ffa')
        plt.ylabel('Fh')
        plt.title(f'ROC Curve for {network}')
        plt.legend(loc="lower right")
        plt.show()

    @staticmethod
    def plot_noise_stats(stats, approach):
        network = "Heteroassociative Multi-Layer Neural Network" if approach == 1 else "Autoassociative Multi-Layer Neural Network"

        x, y, z = [], [], []
        for sdev, (Fh, Ffa) in stats.items():
            x.append(sdev)
            y.append(Fh)
            z.append(Ffa)

        for xe, ye in zip(x, y):
            plt.scatter([xe] * len(ye), ye, marker='.')
        for xe, ze in zip(x, z):
            plt.scatter([xe] * len(ze), ze, marker='+')

        plt.xticks(x)
        plt.axes().set_xscale('log')
        plt.axes().set_xticklabels(x)
        # plt.legend(labels=x)
        plt.title(
            f'Graph of Fh and Ffa for noise-corrupted Alphanumeric Imagery \n (16x16 pixels) for {network}')
        plt.xlabel('Gaussian Noise Level (stdev, at 14 pct xsecn)')
        plt.ylabel('Fh and Ffa')
        plt.show()
