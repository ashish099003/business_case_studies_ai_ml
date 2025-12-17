
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from skimage.color import gray2rgb

def convert_numpy_to_image(pixel_values):

    # 0 = black (background), 255 = white (smiley)


    # Convert to numpy array with uint8
    image_array = np.array(pixel_values, dtype=np.uint8)

    # Display with skimage and matplotlib
    plt.imshow(image_array, cmap='gray')
    plt.title("😊 Winking Smiley (Human View) with skimage")
    plt.axis('off')
    plt.show()

    # Print pixel values (how computer sees it)
    print("🤖 Pixel Values (How Computer Sees It):\n")
    print(image_array)


if __name__=='__main__':
    pixel_values = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 255, 0, 0, 0, 0, 255, 0, 0],  # eyes: left open, right winking
        [0, 0, 255, 0, 0, 0, 0, 0, 0, 0],  # left eye line continues
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # empty line
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # empty line
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # empty line
        [0, 255, 0, 0, 0, 0, 0, 0, 255, 0],  # sides of smile
        [0, 0, 255, 0, 0, 0, 0, 255, 0, 0],  # curved middle
        [0, 0, 0, 255, 255, 255, 255, 0, 0, 0],  # bottom of smile
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    convert_numpy_to_image(pixel_values)
    pixel_values = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 255, 0, 0, 0, 0, 255, 0, 0],  # eyes: left open, right winking
        [0, 0, 255, 0, 0, 0, 0, 0, 0, 0],  # left eye line continues
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # empty line
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # empty line
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # empty line
        [0, 255, 0, 0, 0, 0, 0, 0, 255, 0],  # sides of smile
        [0, 0, 255, 0, 0, 0, 0, 255, 0, 0],  # curved middle
        [0, 0, 0, 255, 255, 255, 255, 0, 0, 0],  # bottom of smile
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    convert_numpy_to_image(pixel_values)
