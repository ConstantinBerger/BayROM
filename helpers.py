from imports import *


def reconstruct_low_res(image_data):
    low_res_micrograph = np.zeros((image_data.shape[0], image_data.shape[0]))
    resolution_factor = int(image_data.shape[1] / image_data.shape[0])

    for x in range(low_res_micrograph.shape[0]):
        for y in range(low_res_micrograph.shape[0]):
            low_res_micrograph[x, y] = np.mean(image_data[x, y * resolution_factor:(y + 1) * resolution_factor])

    return low_res_micrograph


def plot_image(image_array, filename, fov_size, cmap=None, norm=None, save=True):
    fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True, dpi=200)

    if not norm:
        norm = matplotlib.colors.Normalize(vmin=np.min(image_array), vmax=np.max(image_array))

    if not cmap:
        cmap = 'gnuplot2'

    im = ax.imshow(image_array, extent=(0, 1, 1, 0), cmap=cmap, norm=norm)
    # ax.plot((0.05, 0.15), (0.05, 0.05), color='white', linewidth=10, linestyle='-', transform=ax.transAxes)
    fov_size *= 1000
    rect = matplotlib.patches.Rectangle((0.05, 0.05), 0.2, 0.01, edgecolor='none', facecolor='white',
                                        transform=ax.transAxes)
    ax.add_patch(rect)
    rect = matplotlib.patches.Rectangle((0.05, 0.06), 0.2, 0.08, edgecolor='none', facecolor='black',
                                        transform=ax.transAxes)
    ax.add_patch(rect)
    rect = matplotlib.patches.Rectangle((0.05, 0.14), 0.2, 0.01, edgecolor='none', facecolor='white',
                                        transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(0.15, 0.1, str(int(fov_size * 2 / 10)) + 'μm', color='white',
            fontdict=dict(fontsize=15, rotation='horizontal', horizontalalignment='center', verticalalignment='center'),
            transform=ax.transAxes)

    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off')

    if save:
        plt.savefig(filename + '.png', bbox_inches='tight', pad_inches=0, dpi=200)

        file = open(filename + '.obj', 'wb')
        pickle.dump(image_array, file)
        file.close()

    ax.set_title(filename[filename.rfind('/') + 1:])
    plt.show()


def overlay_images(image_array_1, image_array_2, filename, fov_size, save=True):
    fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True, dpi=200)

    norm_rc = matplotlib.colors.Normalize(vmin=np.min(image_array_1), vmax=np.max(image_array_1))
    cmap_rc = LinearSegmentedColormap.from_list('custom', ['black', 'lime'])
    norm_gt = matplotlib.colors.Normalize(vmin=np.min(image_array_2), vmax=np.max(image_array_2))
    cmap_gt = LinearSegmentedColormap.from_list('custom', ['black', 'magenta'])

    im = microplot.microshow(images=[image_array_1, image_array_2], ax=ax, cmaps=[cmap_rc, cmap_gt])
    # ax.plot((0.05, 0.15), (0.05, 0.05), color='white', linewidth=10, linestyle='-', transform=ax.transAxes)
    fov_size *= 1000
    rect = matplotlib.patches.Rectangle((0.05, 0.05), 0.2, 0.01, edgecolor='none', facecolor='white',
                                        transform=ax.transAxes)
    ax.add_patch(rect)
    rect = matplotlib.patches.Rectangle((0.05, 0.06), 0.2, 0.08, edgecolor='none', facecolor='black',
                                        transform=ax.transAxes)
    ax.add_patch(rect)
    rect = matplotlib.patches.Rectangle((0.05, 0.14), 0.2, 0.01, edgecolor='none', facecolor='white',
                                        transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(0.15, 0.1, str(int(fov_size * 2 / 10)) + 'μm', color='white',
            fontdict=dict(fontsize=15, rotation='horizontal', horizontalalignment='center', verticalalignment='center'),
            transform=ax.transAxes)

    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off')

    if save:
        plt.savefig(filename + '.png', bbox_inches='tight', pad_inches=0, dpi=200)

    ax.set_title(filename[filename.rfind('/') + 1:])
    plt.show()
