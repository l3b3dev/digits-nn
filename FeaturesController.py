import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os


class FeaturesController:

    def extract_sift_features(self, image, vector_size=16 * 16):
        alg = cv.SIFT_create()
        # image keypoints
        kps = alg.detect(image)

        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        dsc = dsc.flatten()
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])

        return dsc

    def extract_features(self, images, vector_size=16 * 16):
        features = self.extract_sift_features(images[0], vector_size)
        for i in range(1, images.shape[0]):
            features = np.vstack((features, self.extract_sift_features(images[i], vector_size)))

        return features

    def plot_sift_descriptors(self, path, vec_size=16 * 16):
        fig, (axes1, axes2) = plt.subplots(2, 16)
        fig.suptitle('SIFT extracted features')

        for i, f in enumerate(os.listdir(path)):
            axes = axes1 if i < 16 else axes2
            idx = i if i < 16 else 16-i
            img_name = os.listdir(f'{path}/{f}')
            img = cv.imread(f'{path}/{f}/{img_name[0]}')
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            alg = cv.SIFT_create()
            kp = alg.detect(img)

            kp = sorted(kp, key=lambda x: -x.response)[:vec_size]
            # computing descriptors vector
            kp, dsc = alg.compute(img, kp)

            img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
            axes[idx].grid(False)
            axes[idx].axis('off')
            axes[idx].imshow(img2)

        plt.show()
