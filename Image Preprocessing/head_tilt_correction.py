from PIL import Image
import os 
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from skimage import morphology
import cv2
import math
import imageio


folder = '/media/avcstorage/skull/'
savepath = f'/media/avcstorage/DadosReais_Skull2'

images = []
masked_image = []
rotated_image = []
final_image = []
croped_image = []
threshold = []
final_img = []


def load_images_from_folder(folder):
    global ids
    ids = [os.path.splitext(file)[0] for file in os.listdir(folder)]
    ids = np.array(ids)
    for i in range(len(ids)):
        for filename in os.listdir(f'{folder}{ids[i]}'):
            if filename.endswith(".png"):
                im = Image.open(f'{folder}{ids[i]}/{filename}')
                img = np.array(im)
                images.append(img) 

    return images


def remove_noise(volume):
    for i in range(len(volume)):
        segmentation = morphology.dilation(volume[i, :, :], np.ones((12, 12)))
        labels, label_nb = ndi.label(segmentation)
        label_count = np.bincount(labels.ravel().astype(np.int64))
        label_count[0] = 0
        mask = labels == label_count.argmax()   
        mask = morphology.dilation(mask, np.ones((1, 1)))
        mask = ndi.morphology.binary_fill_holes(mask)
        mask = morphology.dilation(mask, np.ones((3, 3)))  
        masked_img = mask * volume[i, :, :]      
        masked_image.append(masked_img)
        
    return masked_image


def tilt_correction(volume):
    for i in range(len(volume)):
        img = np.uint8(volume[i, :, :]) 
        contours, hier = cv2.findContours (img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # find the biggest contour (c) by the area
        c = max(contours, key = cv2.contourArea)
        (x,y),(MA,ma),angle = cv2.fitEllipse(c)
        #v2.ellipse(img, ((x,y), (MA,ma), angle), color=(255, 255, 0), thickness=2)
        rmajor = max(MA,ma)/2
        if angle > 90:
            angle -= 90
        else:
            angle += 90
        xtop = x + math.cos(math.radians(angle))*rmajor
        ytop = y + math.sin(math.radians(angle))*rmajor
        xbot = x + math.cos(math.radians(angle+180))*rmajor
        ybot = y + math.sin(math.radians(angle+180))*rmajor
        #cv2.line(img, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (255, 255, 0), 3)
        M = cv2.getRotationMatrix2D((x, y), angle-90, 1)  # transformation matrix
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), cv2.INTER_CUBIC)
        rotated_image.append(img)

    return rotated_image


def crop_image(image):
    for i in range(len(image)):
        # Create a mask with the background pixels
        mask = image[i, :, :] == 0
        # Find the brain area
        coords = np.array(np.nonzero(~mask))
        top_left = np.min(coords, axis=1)
        bottom_right = np.max(coords, axis=1)
        # Remove the background
        croped_aux = image[i, top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        croped_image.append(croped_aux)
    
    return croped_image


def add_pad(image, new_height=512, new_width=512):
    # image is a list of arrays
    for i in range(len(image)):
        height, width = image[i].shape[0], image[i].shape[1]
        final_aux = np.zeros((new_height, new_width))
        pad_left = int((new_width - width) // 2)
        pad_top = int((new_height - height) // 2)
        # Replace the pixels with the image's pixels
        final_aux[pad_top:pad_top + height, pad_left:pad_left + width] = image[i]
        save = os.path.join(savepath, ids[i])
        imageio.imwrite(save + '.png', final_aux)
        final_image.append(final_aux)
    
    return final_image


def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


def multi_slice_viewer(volume):
    remove_keymap_conflicts({'q', 'w'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] - volume.shape[0]
    ax.imshow(volume[ax.index], cmap=plt.cm.gray)
    ax.set_title('Computed Tomography (CT)')
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show()

    
def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'q':
        previous_slice(ax)
    elif event.key == 'w':
        next_slice(ax)
    fig.canvas.draw()


def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.set_ylabel('slice %s' % ax.index)
    ax.images[0].set_array(volume[ax.index])


def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.set_ylabel('slice %s' % ax.index)
    ax.images[0].set_array(volume[ax.index])
    

# -------------------------------------------------------------------------- #
# ------------------------------ FUNCTIONS  -------------------------------- #
# -------------------------------------------------------------------------- #

images = load_images_from_folder(folder)
images = np.array(images)

masked_image = remove_noise(images)
masked_image = np.array(masked_image)

rotated_image = tilt_correction(masked_image)
rotated_image = np.array(rotated_image)

croped_image = crop_image(rotated_image)

final_image = add_pad(croped_image)
final_image = np.array(final_image)

multi_slice_viewer(final_image)
