import nibabel as nib
import os
import numpy as np
import cv2
import scipy.ndimage as ndi
import imageio


def window(image, img_min, img_max):
    # WC: Window level - midpoint of the range of the CT numbers
    # WW: Window width - range of the CT numbers
    window_image = image.copy()
    window_image[window_image > img_max] = img_max
    window_image[window_image < img_min] = img_min

    return window_image


def convertInt16touint8(image):
    image=image.astype('int16')
    image=np.uint16(image+np.abs(image.min()))
    image=image.astype('float64')
    image *= (255.0/image.max())

    return image.astype('uint8')


def save_fig(folder, savepath):  # Save as picture
    global ids
    ids = [os.path.splitext(file)[0] for file in os.listdir(folder)]
    ids = np.array(ids)
    for i in range(len(ids)):
        for filename in os.listdir(f'{folder}{ids[i]}'):
            if filename.endswith(".nii"):
                img = nib.load(f'{folder}{ids[i]}/{filename}')
                fdata = img.get_fdata() 
                fdata = np.fliplr(fdata)
                fdata = fdata.T
                ct = window(fdata, -500, 0)  # mask windowing
                ct2 = window(fdata, 200, 450)  # bone windowing 
                ct_o = window(fdata, 0, 80)  # brain windowing
                
                # convert from 16 bits to 8 bits
                ct = convertInt16touint8(ct) 
                ct2 = convertInt16touint8(ct2)
                ct_o = convertInt16touint8(ct_o)
                
                ct2 = ct2 - ct2.min()
                ct2[ct2>0] = 255
                ct[ct>0] = 255
                ct2[ct<255] = 255
                
                ct3 = convertInt16touint8(255-ct2)
                
                pngpath = os.path.join(savepath, ids[i])
                if not os.path.exists(pngpath):
                    os.makedirs(pngpath)
    
                for i in range(len(fdata)):
                    ct3_erode=cv2.morphologyEx(np.uint8(ct3[i,:,:]),cv2.MORPH_ERODE,np.ones((18,18)))
                    labels, label_nb = ndi.label(ct3_erode)
                    label_count = np.bincount(labels.ravel().astype(np.int))
                    label_count[0] = 0
                    mask = labels == label_count.argmax()
                    mask = ndi.morphology.binary_fill_holes(mask)
                    brain_out = ct_o[i,:,:]*mask
                    imageio.imwrite(os.path.join(pngpath,'{}.png'.format(i)), brain_out)


if __name__=='__main__':
    
    folder = '/media/avcstorage/Nii_avc/'
    savepath = f'/media/avcstorage/skull/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    save_fig(folder,savepath)
