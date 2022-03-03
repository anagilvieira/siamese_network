from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from preprocess import dataset

croped_left = []
croped_right = []
image_left = []
image_right = []


def crop(image):
    for i in range(len(image)):
        mask = image[i, :, :] == 0
        coords = np.array(np.nonzero(~mask))
        center = ndimage.center_of_mass(image[i, :, :])
        top_left = np.min(coords, axis=1)
        bottom_right = np.max(coords, axis=1)
        # Remove the background
        croped_aux1 = image[i, top_left[0]:bottom_right[0], top_left[1]:int(center[1])]
        croped_left.append(croped_aux1)
        croped_aux2 = image[i, top_left[0]:bottom_right[0], int(center[1]):bottom_right[1]]
        croped_right.append(croped_aux2)
    return croped_left, croped_right


def padding(image1, image2, new_height=500, new_width=200):
    for i in range(len(image1)):
        height1, width1 = image1[i].shape[0], image1[i].shape[1]
        height2, width2 = image2[i].shape[0], image2[i].shape[1]
        final_aux1 = np.zeros((new_height, new_width))
        final_aux2 = np.zeros((new_height, new_width))
        pad_left1 = int((new_width - width1) // 2)
        pad_top1 = int((new_height - height1) // 2)
        pad_left2 = int((new_width - width2) // 2)
        pad_top2 = int((new_height - height2) // 2)
        # Replace the pixels with the image's pixels
        final_aux1[pad_top1:pad_top1 + height1, pad_left1:pad_left1 + width1] = image1[i]
        image_left.append(final_aux1)
        final_aux2[pad_top2:pad_top2 + height2, pad_left2:pad_left2 + width2] = image2[i]
        image_right.append(final_aux2)
    return image_left, image_right


def similarity(data1, data2):
    for i in range(len(data1)):
        img = (data1[i, :, :])/255.0
        img_fliped = (np.fliplr(data2[i, :, :]))/255.0
        ssim_original = ssim(img, img, data_range=1.0)
        mse_original = mse(img, img)
        cosine_similarity_original = cosine_similarity(img, img)
        #print(cosine_similarity_original)
        ssim_fliped = ssim(img, img_fliped, data_range=1.0)
        mse_fliped = mse(img, img_fliped)
        cosine_similarity_fliped = cosine_similarity(img, img_fliped)
        #print(cosine_similarity_fliped)
    
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharex=True, sharey=True)
        ax = axes.ravel()
        ax[0].imshow(img, cmap=plt.cm.gray)
        ax[0].set_xlabel(f'MSE: {mse_original:.2f}, SSIM: {ssim_original:.2f}')
        ax[0].set_title('Original image')

        ax[1].imshow(img_fliped, cmap=plt.cm.gray)
        ax[1].set_xlabel(f'MSE: {mse_fliped:.2f}, SSIM: {ssim_fliped:.2f}')
        ax[1].set_title('Flipped image')

        f = open("ssim.txt", "a")
        f.write(str(ssim_fliped))
        f.write("\n")
        f.close

        f = open("mse.txt", "a")
        f.write(str(mse_fliped))
        f.write("\n")
        f.close

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    data = dataset.final_image
    croped_left, croped_right = crop(data)
    image_left, image_right = padding(croped_left, croped_right)
    image_left = np.array(image_left)
    image_right = np.array(image_right)
    similarity(image_left, image_right)