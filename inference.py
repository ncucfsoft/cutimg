import os
import sys
import argparse
import numpy as np
from PIL import Image

from tkinter import filedialog
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
sys.path.append(r"D:\python\MODNet-master")
from src.models.modnet import MODNet
import tkinter as tk
from tkinter import messagebox

ckpt_path=r'D:\python\MODNet-master\pretrained\modnet_photographic_portrait_matting.ckpt'
im_name=r'd:\python\person\person.jpg'
outfilename=r'd:\python\person\combgperson.jpg'
backgroundpath=r'd:\python\person\background.jpg'
def resize_image_with_fill(input_image_path, def_width, def_height, fill_color=(255, 255, 255)):
    # 打开图像
    image = Image.open(input_image_path)
    
    # 获取图像的原始宽度和高度
    width, height = image.size
    
    # 计算宽度和高度的缩放比例
    width_ratio = def_width / width
    height_ratio = def_height / height
    
    # 选择更小的比例进行缩放，以保持图像的宽高比
    if width_ratio < height_ratio:
        scale_ratio = width_ratio
    else:
        scale_ratio = height_ratio
        
    # 计算新的缩放后的尺寸
    new_width = int(width * scale_ratio)
    new_height = int(height * scale_ratio)
    
    # 调整图像大小
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # 创建一个新的图像，以便填充
    padded_image = Image.new('RGB', (def_width, def_height), color=fill_color)
    
    # 将调整大小的图像放置在新图像的中心
    offset_x = (new_width - resized_image.width) // 2
    offset_y = (new_height - resized_image.height) // 2
    padded_image.paste(resized_image, (offset_x, offset_y))
    
    # 保存新图像
    return padded_image
im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

def selectimg1():
    global im_name,backgroundpath,outfilename
   
    file_path = filedialog.askopenfilename(title='请选择一个用于扣出人像的图片文件', initialdir='', filetypes=[( "图片文件", ".jpg .png"), ('All Files', ' *')])
    
    if file_path:
        im_name=file_path
      

def selectimg2():
    global im_name,backgroundpath,outfilename
    file_path = filedialog.askopenfilename(title='请选择一个背景图片文件', initialdir='', filetypes=[( "图片文件", ".jpg .png"), ('All Files', ' *')])
    
    if file_path:
        backgroundpath=file_path  

        
def selectimg3():
    global im_name,backgroundpath,outfilename
    file_path = filedialog.asksaveasfile(title='请选择一个用于保存合成结果的图片路径',filetypes = [('jpg file', '*.jpg'), ('所有文件', '*')],initialfile='合成图片文件.jpg',defaultextension='.jpg')
    
    if file_path:
        outfilename=file_path  
        if not '.jpg' in outfilename.name and not '.JPG' in outfilename.name:
         outfilename=file_path.name+'.jpg'
        
def startcreate():
        global im_name,backgroundpath,outfilename
        ref_size = 512

    # define image to tensor transform
    
    # create MODNet and load the pre-trained ckpt
        modnet = MODNet(backbone_pretrained=False)
        modnet = nn.DataParallel(modnet)

        if torch.cuda.is_available():
            modnet = modnet.cuda()
            weights = torch.load(ckpt_path)
        else:
            weights = torch.load(ckpt_path, map_location=torch.device('cpu'))
        modnet.load_state_dict(weights)
        modnet.eval()
        
        # inference images
        #im_names = os.listdir(args.input_path)
        #for im_name in im_names:
        # print('Process image: {0}'.format(im_name))
        # im = Image.open(im_name)
        # max_size = (1280, 960) 
        # width, height = im.size
        # width_factor = max_size[0] / width
        # height_factor = max_size[1] / height
        # factor = min(width_factor, height_factor)

        # # 使用缩放因子计算新的图像尺寸
        # new_size = (int(width * factor), int(height * factor))

        layer1 = Image.open(backgroundpath).convert('RGBA') 
        defwidth=layer1.width
        defheight=layer1.height

        im =resize_image_with_fill(im_name,defwidth,defheight)
        # 调整图像大小
        #im = im.resize(new_size)
    
        curimg=im   
            # unify image channels to 3
        im = np.asarray(im)
        if len(im.shape) == 2:
                im = im[:, :, None]
        if im.shape[2] == 1:
                im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
                im = im[:, :, 0:3]
        im = Image.fromarray(im)
        im = im_transform(im)

        # add mini-batch dim
        im = im[None, :, :, :]

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        # inference
        _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)

        # resize and save matte
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        matte = matte[0][0].data.cpu().numpy()
        matte_name = im_name.split('.')[0] + '.png'

        mask=Image.fromarray((matte * 255).astype('uint8'), mode='L')
        #mask.save(r'd:\python\person\personresult2.png','PNG')
        
        zhezhao =mask
        w,h = curimg.size
        mid_img = Image.new('RGBA', (w, h))
        transparent_image = Image.composite(curimg,mid_img,zhezhao)
    
        final = Image.new("RGBA", layer1.size)             # 合成的image
        final = Image.alpha_composite(final, layer1)
        final = Image.alpha_composite(final, transparent_image)

        final=final.convert('RGB')
        
        final.save(outfilename)
        if hasattr(outfilename,'name'):
            finalstr=outfilename.name
        else:
            finalstr=outfilename

        messagebox.showinfo('生成完毕!' ,'合成图片文件：'+finalstr)

if __name__ == '__main__':
    # define cmd arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input-path', type=str, help='path of input images')
    # parser.add_argument('--output-path', type=str, help='path of output images')
    # parser.add_argument('--ckpt-path', type=str, help='path of pre-trained MODNet')
    # args = parser.parse_args()

    # check input arguments
    # if not os.path.exists(args.input_path):
    #     print('Cannot find input path: {0}'.format(args.input_path))
    #     exit()
    # if not os.path.exists(args.output_path):
    #     print('Cannot find output path: {0}'.format(args.output_path))
    #     exit()
    # if not os.path.exists(args.ckpt_path):
    #     print('Cannot find ckpt path: {0}'.format(args.ckpt_path))
    #     exit()
    root = tk.Tk()
    tk.Button(root, text="选择要抠出人像的图片", command=selectimg1).grid(row=2, column=1, sticky="w", padx=10, pady=5)
    tk.Button(root, text="选择要融合的背景图片", command=selectimg2).grid(row=2, column=2, sticky="e", padx=10, pady=5)
    tk.Button(root, text="选择要保存的合成图片路径", command=selectimg3).grid(row=3, column=1, sticky="e", padx=10, pady=5)
    
    tk.Button(root, text="点击开始", command=startcreate).grid(row=3, column=2, sticky="e", padx=10, pady=5)
    
    width=500
    height=200
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    root.geometry(f"{width}x{height}+{int(x)}+{int(y)}")
    root.title("选择图片或者直接用文件夹下默认的图片开始")

    root.resizable(False,False)
    root.mainloop()
    
    # define hyper-parameters
   

    # transparent_image = curimg.convert('RGBA')
    # pixels = transparent_image.getdata()

    # threshold_image = mask.point(lambda p: 0 if p < 128 else 255)
    # transparent_imagemask = threshold_image.convert('RGBA')
    # pixelsmask = transparent_imagemask.getdata()
   
    

    
    # # 遍历每个像素点，将白色部分设置为透明
    # new_pixels = []
    # for index,pixel in enumerate(pixelsmask):
    #     if pixelsmask[index][:3] == (0, 0, 0):
    #         new_pixels.append((0, 0, 0, 0))
    #     else:
    #         new_pixels.append(pixels[index])
    
    # transparent_image.putdata(new_pixels)

    # read image

        # convert image to PyTorch tensor
        