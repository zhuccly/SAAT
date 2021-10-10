
import menpo.io as mio

def load_images_test(paths, group=None, verbose=True, PLOT=False):
    """Loads and rescales input images to the diagonal of the reference shape.

    Args:
      paths: a list of strings containing the data directories.
      reference_shape (meanshape): a numpy array [num_landmarks, 2]
      group: landmark group containing the grounth truth landmarks.
      verbose: boolean, print debugging info.
    Returns:
      images: a list of numpy arrays containing images.
      shapes: a list of the ground truth landmarks.
      reference_shape (meanshape): a numpy array [num_landmarks, 2].
      shape_gen: PCAModel, a shape generator.
    """
    images = []
    nameList = []

    for path in paths:
        if verbose:
            print('Importing data from {}'.format(path))
        for im in mio.import_images(path, verbose=verbose, as_generator=True):
            nameList.append(str(im.path))
            images.append(im)

    return images