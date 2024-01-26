"""
Stone shape generation using Fourier descriptors.
Switch between four examples by commenting/uncommenting the corresponding lines.
Results are saved in the data folder.

Other parameters:
number of stones: nb_stones
image size: image_size
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from skimage.morphology import binary_closing
from skimage.morphology import disk
from PIL import Image, ImageDraw

image_size = 150
D1=0
D2 = 0.2
D3 = 0.05#0.05
D8= 0.015
nb_stones = 100
def generate_fft_stone_image(image_size,r0,seed,D2 = D2,save_dir = None):
    alpha = -2#Equ12
    beta = -2#Equ12
    D0=r0
    N_harmonic = 64#number of harmonics
    Np = N_harmonic*2#number of points

    def get_Dn(n,alpha,beta,D3,D8):#Equation10-11
        if n==0:
            return D0
        elif n==1:
            return D1
        elif n==2:
            return D2
        elif n==3:
            return D3
        elif n==8:
            return D8
        elif n<8 and n>3:
            return pow(2, alpha*np.log2(n/3)+np.log2(D3))
        elif n>8:
            return pow(2, beta*np.log2(n/8)+np.log2(D8))

    def get_An(Dn,deltan):
        return Dn*np.cos(deltan)

    def get_Bn(Dn,deltan):
        return Dn*np.sin(deltan)


    #compute the amplitude of each harmonic
    Dns = np.zeros((N_harmonic,))
    for i_harmonic in range(1,N_harmonic):
        Dn = get_Dn(i_harmonic,alpha,beta,D3,D8)
        Dns[i_harmonic] = Dn
    #thetan is the phase, a random number between -pi and pi
    np.random.seed(seed)
    theta_n = np.random.uniform(-np.pi,np.pi,(N_harmonic,))
    # compute radius
    r_n = np.zeros((Np,))
    for n in range(Np):
        r_n[n] = r0
        for i_harmonic in range(1,N_harmonic):
            Dn = Dns[i_harmonic]
            theta = n*2*np.pi/Np-np.pi
            r_n[n] += r0*(get_An(Dn,theta_n[i_harmonic])*np.cos(i_harmonic*theta)+get_Bn(Dn,theta_n[i_harmonic])*np.sin(i_harmonic*theta))

    def convert_radius(radius):
        #convert radius from polar coordinate to cartesian coordinate
        angles = np.arange(0, 2*np.pi, 2*np.pi/Np)
        radius = radius
        #element wise multiplication
        xs = np.multiply(radius,np.cos(angles))
        ys = np.multiply(radius,np.sin(angles))
        return xs,ys
    xs,ys = convert_radius(r_n)
    #create image from polygon points
    # Create a blank canvas
    canvas = Image.new('L', (image_size, image_size), 0)
    # Draw a polygon
    draw = ImageDraw.Draw(canvas)
    points = []
    for i in range(len(xs)):
        points.append((xs[i]+image_size/2,ys[i]+image_size/2))
    draw.polygon(points, fill=255)
    #closing holes
    
    canvas = np.asarray(canvas)
    canvas = binary_closing(canvas, disk(2))
    #save image
    cv2.imwrite(save_dir+f"stone_{seed}.png",(canvas*255).astype(np.uint8))


save_dir = "./data/D2_0/"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
for i in range(nb_stones):
    generate_fft_stone_image(image_size,30,i,0,save_dir)


# save_dir = "./data/D2_0_uniformr/"
# for i in range(nb_stones):
#     np.random.seed(i)
#     radius = np.random.randint(10,50)
#     generate_fft_stone_image(image_size,radius,i,0,save_dir)

# save_dir = "./data/D2_02/"
# for i in range(nb_stones):
#     radius = 30
#     generate_fft_stone_image(image_size,radius,i,0.2,save_dir)

# save_dir = "./data/D2_02_uniformr/"
# for i in range(nb_stones):
#     np.random.seed(i)
#     radius = np.random.randint(10,50)
#     generate_fft_stone_image(image_size,radius,i,save_dir)

