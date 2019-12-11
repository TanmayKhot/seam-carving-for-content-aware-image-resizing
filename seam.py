import sys

import numpy as np
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve
from skimage import color


def conv(image,kernel):
	#Initializing the dimensional variables
	Hi, Wi = image.shape
	Hk, Wk = kernel.shape
	out = np.zeros((Hi,Wi))

	#Using edge padding since zero padding will give large derivates at the borders
	pad_height = Hk // 2
	pad_width = Wk // 2
	padding = ((pad_height,pad_height) , (pad_width,pad_width))
	padded = np.pad(image, padding, mode = 'edge')

	#Performing convolution
	kernel = np.flip(np.flip(kernel,0),1)
	for i in range(Hi):
		for j in range(Wi):
			out[i,j]= sum(sum(kernel*padded[i:i+Hk,j:j+Wk]))
    
	return out

def energy(image):
	# e(I) = dx(I) + dy(I)
	dxkernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
	dykernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
	dxI = conv(image,dxkernel)
	dyI = conv(image,dykernel)
	E = np.add(dxI,dyI)

	return E


def minimum_seam(img):
    r, c = img.shape
    energy_map = energy(img)

    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, r):
        for j in range(0, c):
            # Handle the left edge of the image, to ensure we don't index -1
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i - 1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    return M, backtrack

def carve_column(img):
    r, c = img.shape

    M, backtrack = minimum_seam(img)

    # Create a (r, c) matrix filled with the value True
    # We'll be removing all pixels from the image which
    # have False later
    mask = np.ones((r, c), dtype=np.bool)

    # Find the position of the smallest element in the
    # last row of M
    j = np.argmin(M[-1])

    for i in reversed(range(r)):
        # Mark the pixels for deletion
        mask[i, j] = False
        j = backtrack[i, j]

    # Since the image has 3 channels, we convert our
    # mask to 3D
    # ++++ mask = np.stack([mask] * 3, axis=2)

    # Delete all the pixels marked False in the mask,
    # and resize it to the new image dimensions
    img = img[mask].reshape((r, c - 1))

    return img

def crop_c(img, scale_c):
    r, c = img.shape
    new_c = int(scale_c * c)

    for i in range(c - new_c): # use range if you don't want to use tqdm
        img = carve_column(img)

    return img

#Logic for rows will be same as columns. Hence we rotate the image
#by 90 degrees and pass it to the column function and again rotate by 90 deg 3 more times 

def crop_r(img, scale_r):
    img = np.rot90(img, 1, (0, 1))
    img = crop_c(img, scale_r)
    img = np.rot90(img, 3, (0, 1))
    return img

def main():

    colrow = sys.argv[1]
    scale = float(sys.argv[2])
    inputimage = sys.argv[3]
    outputimage = sys.argv[4]
    #deriv = sys.argv[5]

    inputimg = imread(inputimage)
    img = color.rgb2gray(inputimg)

    #imwrite(energy(img),deriv)


    if colrow == 'r':
        output = crop_r(img, scale)
    elif colrow == 'c':
        output = crop_c(img, scale)
    
    imwrite(outputimg, output)

if __name__ == '__main__':
    main()
