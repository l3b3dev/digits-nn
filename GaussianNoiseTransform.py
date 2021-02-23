import torch


# Noise is Gaussian with 10 percent cross-section
class GaussianNoiseTransform(object):
    def __init__(self, mean=0., std=1., k=25):
        self.std = std
        self.mean = mean
        self.k = k

    def __call__(self, tensor):
        # reshape and flatten
        x_transf = torch.flatten(tensor[0, :], start_dim=0)

        n = x_transf.size(0)
        perm = torch.randperm(n)
        idx = perm[:(n - self.k)]

        noise = torch.randn(x_transf.size())
        # only 10% is noise
        noise[idx] = 0.

        corrupted_image = x_transf + noise * self.std + self.mean
        # renormalize
        corrupted_image -= corrupted_image.min(0, keepdim=True)[0]
        corrupted_image /= corrupted_image.max(0, keepdim=True)[0]

        tensor[0, :] = corrupted_image.reshape(16, 16)

        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
