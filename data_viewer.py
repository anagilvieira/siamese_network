import matplotlib.pyplot as plt
import numpy as np
from preprocess import dataset


def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


# Permite correr as várias slices no mesmo gráfico
def multi_slice_viewer(volume):
    remove_keymap_conflicts({'q', 'w'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] - volume.shape[0]
    ax.imshow(volume[ax.index], cmap=plt.cm.gray)
    ax.set_title('Computed Tomography (CT)')
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show(block=False)

    
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


if __name__ == '__main__':
    data = dataset.final_image
    print(data.shape)
    multi_slice_viewer(data)