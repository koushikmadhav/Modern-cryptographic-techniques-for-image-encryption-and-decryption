

#################encryption projct heads
import os
import math
import numpy as np
from utils import *
from scipy import fftpack


from huffman import HuffmanTree
from pyblake2 import blake2b
import random

import pickle


#####################
from tkinter import Label,Tk
import tkinter.filedialog
from tkinter import *
from tkinter import messagebox as ms


from tkinter import *
from PIL import ImageTk
from PIL import Image

def main():
    root=Tk()
    
    
    
    
    
    
    label=Label(root, text="545",fg='white',bg='white',height=16,width=40,font=("bold", 10),borderwidth=1,relief="ridge")
    label.place(x=85,y=20)
    
    
    label_0 = Label(root, text="Login form",bg='RoyalBlue4',fg='white',height=2,width=20,font=("bold", 20))
    label_0.place(x=83,y=20)
    
    
    label_6 = Label(root, text="Please login to continue",fg='white',bg='RoyalBlue4',width=25,font=("bold", 8))
    label_6.place(x=180,y=70)
    
    label_1 = Label(root, text="user name",fg='grey',bg='white',width=10,font=("bold", 10))
    label_1.place(x=123,y=126)
    
    username = Entry(root,width=25,bg='thistle2')
    username.place(x=200,y=130)
    
    
    label_2 = Label(root, text="password",bg='white',fg='grey',width=8,font=("bold", 10))
    label_2.place(x=133,y=176)
    
    password = Entry(root,width=25,bg='thistle2')
    password.place(x=200,y=180)

    
                
        
    Button(root, text='login',width=15,bg='RoyalBlue4',fg='white',command=lambda:login(root,username,password)).place(x=240,y=230)
    Button(root, text='Exit',width=10,bg='brown4',fg='white',command=lambda:root.destroy()).place(x=158,y=230)
    window1=windows(root,"login","Login")
    return None






##########################################################################
def login(root,username,password):
    	#Establish Connection
       
        root1=root
        uname=username.get()
        upass=password.get()
        if uname=="root" and upass=="12345":
            ms.showinfo('Login','Login succeed....')
            choose(root1)
            
     
        else:
            ms.showerror('Oops!','Login failed.')
            return NONE
            

#########       2nd WINDOW  #############################################################
def choose(root1):
    a=root1
    a.destroy()
    root = Tk()
    
    
    
    Label(root, width=25,height=15,bg='lavender',borderwidth=2,relief="groove").place(x=70,y=30)
    Label(root, width=25,height=15,bg='lavender',borderwidth=2,relief="groove").place(x=240,y=30)
    
    image=Image.open("lock.png")
    image=image.resize((109,100),Image.ANTIALIAS)
    img=ImageTk.PhotoImage(image)
    Label(root, image=img).place(x=100,y=100)
    
    image2=Image.open("unlock.png")
    image2=image2.resize((109,100),Image.ANTIALIAS)
    img2=ImageTk.PhotoImage(image2)
    Label(root, image=img2).place(x=270,y=100)
        
    
    
    
    label_0 = Label(root, text="Choose your operation",bg='RoyalBlue4',fg='white',width=20,height=0,font=( 12))
    label_0.place(x=0,y=0)
    Button(root, text='Encryption',width=20,bg='RoyalBlue4',fg='white',command=lambda:encryption(root)).place(x=80,y=200)
    Button(root, text='Decryption',width=20,bg='RoyalBlue4',fg='white',command=lambda:decryption(root)).place(x=250,y=200)
    window2=windows(root,"Encryption|decryption","Choose your operation") 
############################################################################################################
########################################Decryption part##################################################
    #####################################################################
def decryption(root1):
        path=None
        
        
        def filechooser(root):
            def mouseClick(event):
                image = Image.open(path)
                image.show()
            
            nonlocal path
            path=None
            

            
            path=filedialog.askopenfilename(filetypes=[("Image File",'.jpg'),("Image File",'.jpeg'),("Image File",'.png')])
            
            disp=Image.open(path)
            disp=disp.resize((128,128),Image.ANTIALIAS)
            disp1=ImageTk.PhotoImage(disp)
            lab=Label(root, image=disp1,borderwidth=2,relief='groove')
            lab.image=disp1
            lab.place(x=290,y=115)
            selected.place(x=273,y=95)
            lab.bind("<Button>",mouseClick)
        
        
        def decrypt():
            #######################Decrypt body######################
            
            keyval1=key23.get()
            
            
            
            
            
            
            
            
            
            
            if not keyval1:
                ms.showerror('error','key is missing!!!')
            elif path==None:
                ms.showerror('error','encrypted image is missing')
            else:
                keyval1=int(keyval1)
                def mapkey2(key,j):
                    while key>=j:
                        key=key/j
                    return round(key)
                
                
                def Decrypt(the_list,key2,r):
                    i,j,k=the_list.shape
                    key1=the_list[key2][0][0]
                    the_list=the_list+r
                    for a in range(i):
                        for b in range(j): 
                            for c in range(k):
                                the_list[key2][b][c]=the_list[key2][b][c]-key1
                    the_list=the_list+key1      
                    return the_list
                
                
                def zigzag_points(rows, cols):
                    UP, DOWN, RIGHT, LEFT, UP_RIGHT, DOWN_LEFT = range(6)
                    def move(direction, point):
                        return {
                            UP: lambda point: (point[0] - 1, point[1]),
                            DOWN: lambda point: (point[0] + 1, point[1]),
                            LEFT: lambda point: (point[0], point[1] - 1),
                            RIGHT: lambda point: (point[0], point[1] + 1),
                            UP_RIGHT: lambda point: move(UP, move(RIGHT, point)),
                            DOWN_LEFT: lambda point: move(DOWN, move(LEFT, point))
                        }[direction](point)
                    def inbounds(point):
                        return 0 <= point[0] < rows and 0 <= point[1] < cols
                    point = (0, 0)
                    move_up = True
                    for i in range(rows * cols):
                        yield point
                        if move_up:
                            if inbounds(move(UP_RIGHT, point)):
                                point = move(UP_RIGHT, point)
                            else:
                                move_up = False
                                if inbounds(move(RIGHT, point)):
                                    point = move(RIGHT, point)
                                else:
                                    point = move(DOWN, point)
                        else:
                            if inbounds(move(DOWN_LEFT, point)):
                                point = move(DOWN_LEFT, point)
                            else:
                                move_up = True
                                if inbounds(move(DOWN, point)):
                                    point = move(DOWN, point)
                                else:
                                    point = move(RIGHT, point)
                                    
                
                def zigzag_to_block(zigzag):
                    rows = cols = int(math.sqrt(len(zigzag)))
                    if rows * cols != len(zigzag):
                        raise ValueError("length of zigzag should be a perfect square")
                    block = np.empty((rows, cols), np.int32)
                
                    for i, point in enumerate(zigzag_points(rows, cols)):
                        block[point] = zigzag[i]
                
                    return block
                
                
                def load_quantization_table(component):
                    if component == 'lum':
                        q = np.array([[2, 2, 2, 2, 3, 4, 5, 6],
                                      [2, 2, 2, 2, 3, 4, 5, 6],
                                      [2, 2, 2, 2, 4, 5, 7, 9],
                                      [2, 2, 2, 4, 5, 7, 9, 12],
                                      [3, 3, 4, 5, 8, 10, 12, 12],
                                      [4, 4, 5, 7, 10, 12, 12, 12],
                                      [5, 5, 7, 9, 12, 12, 12, 12],
                                      [6, 6, 9, 12, 12, 12, 12, 12]])
                    elif component == 'chrom':
                        q = np.array([[3, 3, 5, 9, 13, 15, 15, 15],
                                      [3, 4, 6, 11, 14, 12, 12, 12],
                                      [5, 6, 9, 14, 12, 12, 12, 12],
                                      [9, 11, 14, 12, 12, 12, 12, 12],
                                      [13, 14, 12, 12, 12, 12, 12, 12],
                                      [15, 12, 12, 12, 12, 12, 12, 12],
                                      [15, 12, 12, 12, 12, 12, 12, 12],
                                      [15, 12, 12, 12, 12, 12, 12, 12]])
                    else:
                        raise ValueError((
                            "component should be either 'lum' or 'chrom', "
                            "but '{comp}' was found").format(comp=component))
                    return q
                
                
                def dequantize(block, component):
                    q = load_quantization_table(component)
                    return block * q
                
                
                def idct_2d(image):
                    return fftpack.idct(fftpack.idct(image.T, norm='ortho').T, norm='ortho')
                
                
                
                if __name__ == "__main__":
                    file=open("bQ","rb")
                    bitstream=pickle.load(file)
                    dc=bitstream[0]
                    ac=bitstream[1]  
                    blocks_count=bitstream[2]
                    tables=bitstream[3]
                    r=bitstream[4]
                    key1=bitstream[5]
                    key2R=bitstream[6]
                    print("Key_Encryption1 =",key1)
                    i,j,k=ac.shape
                    key2=keyval1
                    if(key2R!=key2):
                        r=np.random.randint(0, 100, size=(i, j, k))
                    key2m=mapkey2(key2,j)
                    print("Key_Encryption2 =",key2m)
                    ac=Decrypt(ac,key2m,r)
                    block_side = 8
                    image_side = int(math.sqrt(blocks_count)) * block_side
                    blocks_per_line = image_side // block_side
                    npmat = np.empty((image_side, image_side, 3), dtype=np.uint8)
                    for block_index in range(blocks_count):
                        i = block_index // blocks_per_line * block_side
                        j = block_index % blocks_per_line * block_side
                        for c in range(3):
                            zigzag = [dc[block_index, c]] + list(ac[block_index, :, c])
                            quant_matrix = zigzag_to_block(zigzag)
                            dct_matrix = dequantize(quant_matrix, 'lum' if c == 0 else 'chrom')
                            block = idct_2d(dct_matrix)
                            npmat[i:i+8, j:j+8, c] = block + 128
                    image = Image.fromarray(npmat, 'YCbCr')
                    image = image.convert('RGB')
                    image.save("decryptedimg.jpg","JPEG")
                    
                    
                    disp=Image.open("decryptedimg.jpg")
                    disp=disp.resize((128,128),Image.ANTIALIAS)
                    disp1=ImageTk.PhotoImage(disp)
                    
                    def mouseClick(event):
                        image = Image.open("decryptedimg.jpg")
                        image.show()
                    
                    lab=Label(root, image=disp1,borderwidth=2,relief='groove')
                    lab.image=disp1
                    lab.place(x=290,y=115)
                    lab.bind("<Button>",mouseClick)
                    selected.configure(text="Decrypted image:")
                    selected.place(x=273,y=95)
                    
                    ms.showinfo('Success!','Decryption completed')
                
                
            
            

        root1.destroy()
        root = Tk()
        
        Label(root, width=50,height=15,bg='grey94',borderwidth=2,relief="groove").place(x=70,y=30)
        # Label(root, width=25,height=15,bg='lavender',borderwidth=2,relief="groove").place(x=240,y=30)
        
        image=Image.open("decryption.png")
        image=image.resize((50,50),Image.ANTIALIAS)
        img=ImageTk.PhotoImage(image)
        Label(root, image=img).place(x=230,y=40)
        
        
        
        
        image12=Image.open("key.png")
        image12=image12.resize((20,20),Image.ANTIALIAS)
        img12=ImageTk.PhotoImage(image12)
        Label(root, image=img12).place(x=99,y=134)
        
        key2 = Label(root, text="Decryption key:",fg='grey',bg='grey94',width=11,font=("bold", 10))
        key2.place(x=120,y=136)
        
        key23 = Entry(root,width=9,bg='thistle2')
        key23.place(x=215,y=140)
        
        
        
        
        image1=Image.open("find.png")
        image1=image1.resize((20,20),Image.ANTIALIAS)
        img1=ImageTk.PhotoImage(image1)
        Label(root, image=img1).place(x=97,y=178)
        
        label_0 = Label(root, text="Select image:",fg='grey',bg='grey94',width=9,font=("bold",10))
        label_0.place(x=120,y=180)
        
        Button(root, text='Browse',width=7,height=0,command=lambda:filechooser(root)).place(x=215,y=180)
        
        
        selected = Label(root, text="Selected image:",fg='grey',bg='grey94',width=16,font=("bold",8))
        selected.place_forget #(x=273,y=95)
        
        
        
       
        
        Button(root, text='Back',width=10,bg='SpringGreen4',fg='white',command=lambda:choose(root)).place(x=90,y=220)
        Button(root, text='Decrypt',width=15,bg='brown',fg='white',command=decrypt).place(x=170,y=220)
        window2=windows(root,"Encryption"," ")
##############################################################################
#################################End of decryption####################################
#####################################################################################

#######################################################################
########----ENCRYPTION PART--------------############      
############ENCRYPTION WINDOW#################


def encryption(root1):
        path=None
        
        
        def filechooser(root):
            nonlocal path
            print(path)
            def mouseClick(event):
                        image = Image.open(path)
                        image.show()
            
              
           
            
            path=filedialog.askopenfilename(filetypes=[("Image File",'.jpg')])
            
            disp=Image.open(path)
            disp=disp.resize((128,128),Image.ANTIALIAS)
            disp1=ImageTk.PhotoImage(disp)
            
            lab=Label(root, image=disp1,borderwidth=2,relief='groove')
            lab.image=disp1
            lab.place(x=290,y=115)
            lab.bind("<Button>",mouseClick)  
            selected.place(x=273,y=95)
            
            
        def encrypt():
            keyval=key22.get()
            if not keyval:
                ms.showerror('Oops!','key is required.')
            elif path==None:
                ms.showerror('Oops!','image is not found.')
                
            else:
                keyval=int(keyval)
            ############Encryption code ********************

                def quantize(block, component):
                    q = load_quantization_table(component)
                    return (block / q).round().astype(np.int32)
                
                
                def block_to_zigzag(block):
                    return np.array([block[point] for point in zigzag_points(*block.shape)])
                
                
                def dct_2d(image):
                    return fftpack.dct(fftpack.dct(image.T, norm='ortho').T, norm='ortho')
                
                
                def run_length_encode(arr):
                    last_nonzero = -1
                    for i, elem in enumerate(arr):
                        if elem != 0:
                            last_nonzero = i
                
                    symbols = []
                    values = []
                
                    run_length = 0
                
                    for i, elem in enumerate(arr):
                        if i > last_nonzero:
                            symbols.append((0, 0))
                            values.append(int_to_binstr(0))
                            break
                        elif elem == 0 and run_length < 15:
                            run_length += 1
                        else:
                            size = bits_required(elem)
                            symbols.append((run_length, size))
                            values.append(int_to_binstr(elem))
                            run_length = 0
                    return symbols, values
                
                
                def write_to_file(filepath, dc, ac, blocks_count, tables):
                    try:
                        f = open(filepath, 'w')
                    except FileNotFoundError as e:
                        raise FileNotFoundError(
                                "No such directory: {}".format(
                                    os.path.dirname(filepath))) from e
                
                    for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:
                        f.write(uint_to_binstr(len(tables[table_name]), 16))
                
                        for key, value in tables[table_name].items():
                            if table_name in {'dc_y', 'dc_c'}:
                                f.write(uint_to_binstr(key, 4))
                                f.write(uint_to_binstr(len(value), 4))
                                f.write(value)
                            else:
                                f.write(uint_to_binstr(key[0], 4))
                                f.write(uint_to_binstr(key[1], 4))
                                f.write(uint_to_binstr(len(value), 8))
                                f.write(value)
                
                    f.write(uint_to_binstr(blocks_count, 32))
                
                    for b in range(blocks_count):
                        for c in range(3):
                            category = bits_required(dc[b, c])
                            symbols, values = run_length_encode(ac[b, :, c])
                            dc_table = tables['dc_y'] if c == 0 else tables['dc_c']
                            ac_table = tables['ac_y'] if c == 0 else tables['ac_c']
                
                            f.write(dc_table[category])
                            f.write(int_to_binstr(dc[b, c]))
                            
                            
                            for i in range(len(symbols)):
                                f.write(ac_table[tuple(symbols[i])])
                                f.write(values[i])
                    f.close()
                
                
                def encoder():
                    output_file = 'a.txt'
                
                    image = Image.open(path)
                    
                    ycbcr = image.convert('YCbCr')
                
                    npmat = np.array(ycbcr, dtype=np.uint8)
                
                    rows, cols = npmat.shape[0], npmat.shape[1]
                
                    if rows % 8 == cols % 8 == 0:
                        blocks_count = rows // 8 * cols // 8
                    else:
                        raise ValueError(("the width and height of the image "
                                          "should both be mutiples of 8"))
                    dc = np.empty((blocks_count, 3), dtype=np.int32)
                    ac = np.empty((blocks_count, 63, 3), dtype=np.int32)
                
                    for i in range(0, rows, 8):
                        for j in range(0, cols, 8):
                            try:
                                block_index += 1
                            except NameError:
                                block_index = 0
                
                            for k in range(3):
                                block = npmat[i:i+8, j:j+8, k] - 128
                
                                dct_matrix = dct_2d(block)
                                quant_matrix = quantize(dct_matrix,
                                                        'lum' if k == 0 else 'chrom')
                                zz = block_to_zigzag(quant_matrix)
                
                                dc[block_index, k] = zz[0]
                                ac[block_index, :, k] = zz[1:]
                
                    H_DC_Y = HuffmanTree(np.vectorize(bits_required)(dc[:, 0]))
                    H_DC_C = HuffmanTree(np.vectorize(bits_required)(dc[:, 1:].flat))
                    H_AC_Y = HuffmanTree(
                            flatten(run_length_encode(ac[i, :, 0])[0]
                                    for i in range(blocks_count)))
                    H_AC_C = HuffmanTree(
                            flatten(run_length_encode(ac[i, :, j])[0]
                                    for i in range(blocks_count) for j in [1, 2]))
                
                    tables = {'dc_y': H_DC_Y.value_to_bitstring_table(),
                              'ac_y': H_AC_Y.value_to_bitstring_table(),
                              'dc_c': H_DC_C.value_to_bitstring_table(),
                              'ac_c': H_AC_C.value_to_bitstring_table()}   
                    return dc, ac, blocks_count, tables
                    
                
                def en(the_list,key1,key2):
                    i,j,k=the_list.shape
                    global randA
                    randA = np.random.randint(0, 100, size=(i, j, k))
                    the_list=the_list-key1
                    for a in range(i):
                        for b in range(j):
                            for c in range(k):
                                
                                the_list[key2][b][c]=the_list[key2][b][c]+key1
                                
                
                    the_list=the_list-randA
                    the_list[key2][0][0]=key1
                           
                    return the_list
                
                
                def load_quantization_table(component):
                    if component == 'lum':
                        q = np.array([[2, 2, 2, 2, 3, 4, 5, 6],
                                      [2, 2, 2, 2, 3, 4, 5, 6],
                                      [2, 2, 2, 2, 4, 5, 7, 9],
                                      [2, 2, 2, 4, 5, 7, 9, 12],
                                      [3, 3, 4, 5, 8, 10, 12, 12],
                                      [4, 4, 5, 7, 10, 12, 12, 12],
                                      [5, 5, 7, 9, 12, 12, 12, 12],
                                      [6, 6, 9, 12, 12, 12, 12, 12]])
                    elif component == 'chrom':
                        q = np.array([[3, 3, 5, 9, 13, 15, 15, 15],
                                      [3, 4, 6, 11, 14, 12, 12, 12],
                                      [5, 6, 9, 14, 12, 12, 12, 12],
                                      [9, 11, 14, 12, 12, 12, 12, 12],
                                      [13, 14, 12, 12, 12, 12, 12, 12],
                                      [15, 12, 12, 12, 12, 12, 12, 12],
                                      [15, 12, 12, 12, 12, 12, 12, 12],
                                      [15, 12, 12, 12, 12, 12, 12, 12]])
                    else:
                        raise ValueError((
                            "component should be either 'lum' or 'chrom', "
                            "but '{comp}' was found").format(comp=component))
                
                    return q
                
                
                def zigzag_points(rows, cols):
                    UP, DOWN, RIGHT, LEFT, UP_RIGHT, DOWN_LEFT = range(6)
                
                    def move(direction, point):
                        return {
                            UP: lambda point: (point[0] - 1, point[1]),
                            DOWN: lambda point: (point[0] + 1, point[1]),
                            LEFT: lambda point: (point[0], point[1] - 1),
                            RIGHT: lambda point: (point[0], point[1] + 1),
                            UP_RIGHT: lambda point: move(UP, move(RIGHT, point)),
                            DOWN_LEFT: lambda point: move(DOWN, move(LEFT, point))
                        }[direction](point)
                
                    def inbounds(point):
                        return 0 <= point[0] < rows and 0 <= point[1] < cols
                    point = (0, 0)
                    move_up = True
                
                    for i in range(rows * cols):
                        yield point
                        if move_up:
                            if inbounds(move(UP_RIGHT, point)):
                                point = move(UP_RIGHT, point)
                            else:
                                move_up = False
                                if inbounds(move(RIGHT, point)):
                                    point = move(RIGHT, point)
                                else:
                                    point = move(DOWN, point)
                        else:
                            if inbounds(move(DOWN_LEFT, point)):
                                point = move(DOWN_LEFT, point)
                            else:
                                move_up = True
                                if inbounds(move(DOWN, point)):
                                    point = move(DOWN, point)
                                else:
                                    point = move(RIGHT, point)
                
                
                def bits_required(n):
                    n = abs(n)
                    result = 0
                    while n > 0:
                        n >>= 1
                        result += 1
                    return result
                
                
                def binstr_flip(binstr):
                    if not set(binstr).issubset('01'):
                        raise ValueError("binstr should have only '0's and '1's")
                    return ''.join(map(lambda c: '0' if c == '1' else '1', binstr))
                
                
                def uint_to_binstr(number, size):
                    return bin(number)[2:][-size:].zfill(size)
                
                
                def int_to_binstr(n):
                    if n == 0:
                        return ''
                
                    binstr = bin(abs(n))[2:]
                    return binstr if n > 0 else binstr_flip(binstr)
                
                
                def flatten(lst):
                    return [item for sublist in lst for item in sublist]
                
                
                class JPEGFileReader:
                    TABLE_SIZE_BITS = 16
                    BLOCKS_COUNT_BITS = 32
                
                    DC_CODE_LENGTH_BITS = 4
                    CATEGORY_BITS = 4
                
                    AC_CODE_LENGTH_BITS = 8
                    RUN_LENGTH_BITS = 4
                    SIZE_BITS = 4
                
                    def __init__(self, filepath):
                        self.__file = open(filepath, 'r')
                
                    def read_int(self, size):
                        if size == 0:
                            return 0
                        bin_num = self.__read_str(size)
                        if bin_num[0] == '1':
                            return self.__int2(bin_num)
                        else:
                            return self.__int2(binstr_flip(bin_num)) * -1
                
                    def read_dc_table(self):
                        table = dict()
                
                        table_size = self.__read_uint(self.TABLE_SIZE_BITS)
                        for _ in range(table_size):
                            category = self.__read_uint(self.CATEGORY_BITS)
                            code_length = self.__read_uint(self.DC_CODE_LENGTH_BITS)
                            code = self.__read_str(code_length)
                            table[code] = category
                        return table
                
                    def read_ac_table(self):
                        table = dict()
                
                        table_size = self.__read_uint(self.TABLE_SIZE_BITS)
                        for _ in range(table_size):
                            run_length = self.__read_uint(self.RUN_LENGTH_BITS)
                            size = self.__read_uint(self.SIZE_BITS)
                            code_length = self.__read_uint(self.AC_CODE_LENGTH_BITS)
                            code = self.__read_str(code_length)
                            table[code] = (run_length, size)
                        return table
                
                    def read_blocks_count(self):
                        return self.__read_uint(self.BLOCKS_COUNT_BITS)
                
                    def read_huffman_code(self, table):
                        prefix = ''
                        while prefix not in table:
                            prefix += self.__read_char()
                        return table[prefix]
                
                    def __read_uint(self, size):
                        if size <= 0:
                            raise ValueError("size of unsigned int should be greater than 0")
                        return self.__int2(self.__read_str(size))
                
                    def __read_str(self, length):
                        return self.__file.read(length)
                
                    def __read_char(self):
                        return self.__read_str(1)
                
                    def __int2(self, bin_num):
                        return int(bin_num, 2)
                
                
                def read_image_file(filepath):
                    reader = JPEGFileReader(filepath)
                
                    tables = dict()
                    for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:
                        if 'dc' in table_name:
                            tables[table_name] = reader.read_dc_table()
                        else:
                            tables[table_name] = reader.read_ac_table()
                
                    blocks_count = reader.read_blocks_count()
                
                    dc = np.empty((blocks_count, 3), dtype=np.int32)
                    ac = np.empty((blocks_count, 63, 3), dtype=np.int32)
                
                    for block_index in range(blocks_count):
                        for component in range(3):
                            dc_table = tables['dc_y'] if component == 0 else tables['dc_c']
                            ac_table = tables['ac_y'] if component == 0 else tables['ac_c']
                
                            category = reader.read_huffman_code(dc_table)
                            dc[block_index, component] = reader.read_int(category)
                
                            cells_count = 0
                
                            while cells_count < 63:
                                run_length, size = reader.read_huffman_code(ac_table)
                
                                if (run_length, size) == (0, 0):
                                    while cells_count < 63:
                                        ac[block_index, cells_count, component] = 0
                                        cells_count += 1
                                else:
                                    for i in range(run_length):
                                        ac[block_index, cells_count, component] = 0
                                        cells_count += 1
                                    if size == 0:
                                        ac[block_index, cells_count, component] = 0
                                    else:
                                        value = reader.read_int(size)
                                        ac[block_index, cells_count, component] = value
                                    cells_count += 1    
                    return dc, ac, blocks_count,tables
                
                
                def zigzag_to_block(zigzag):
                    rows = cols = int(math.sqrt(len(zigzag)))
                
                    if rows * cols != len(zigzag):
                        raise ValueError("length of zigzag should be a perfect square")
                
                    block = np.empty((rows, cols), np.int32)
                
                    for i, point in enumerate(zigzag_points(rows, cols)):
                        block[point] = zigzag[i]
                
                    return block
                
                
                def dequantize(block, component):
                    q = load_quantization_table(component)
                    return block * q
                
                
                def idct_2d(image):
                    return fftpack.idct(fftpack.idct(image.T, norm='ortho').T, norm='ortho')
                
                def map255(key):
                    if key>255:
                        key=key/255
                        map255(key)
                    return round(key)
                
                def mapkey2(key,j):
                    while key>=j:
                        key=key/j
                    return round(key)
                
                def decoder(key1,dc, ac, blocks_count, tables,key2):
                    i,j,k=ac.shape
                    
                    key2m=mapkey2(key2,j)
                    ac=en(ac,key1,key2m)
                    
                    block_side = 8
                
                    image_side = int(math.sqrt(blocks_count)) * block_side
                
                    blocks_per_line = image_side // block_side
                
                    npmat = np.empty((image_side, image_side, 3), dtype=np.uint8)
                
                    for block_index in range(blocks_count):
                        i = block_index // blocks_per_line * block_side
                        j = block_index % blocks_per_line * block_side
                
                        for c in range(3):
                            zigzag = [dc[block_index, c]] + list(ac[block_index, :, c])
                            quant_matrix = zigzag_to_block(zigzag)
                            dct_matrix = dequantize(quant_matrix, 'lum' if c == 0 else 'chrom')
                            block = idct_2d(dct_matrix)
                            npmat[i:i+8, j:j+8, c] = block + 128
                            
                
                    image = Image.fromarray(npmat, 'YCbCr')
                    image = image.convert('RGB')
                    image.save("encrypted_image.jpg","JPEG")
                    
                    
                    def mouseClick(event):
                        image = Image.open("encrypted_image.jpg")
                        image.show()
                    
                    
                    
                    disp=Image.open("encrypted_image.jpg")
                    disp=disp.resize((128,128),Image.ANTIALIAS)
                    disp1=ImageTk.PhotoImage(disp)
                    
                    
                    
                    lab=Label(root, image=disp1,borderwidth=2,relief='groove')
                    lab.image=disp1
                    lab.place(x=290,y=115)
                    lab.bind("<Button>",mouseClick)
                    selected.configure(text="Encrypted form")
                    selected.place(x=273,y=95)
                    
                    
                    
                    
                    
                    
                    
                    return dc, ac, blocks_count, tables
                
                
                
                if __name__ == "__main__":
                    with open(path,'rb') as picture:
                        d=picture.read()
                        b=bytes(d)
                        h = blake2b()
                        h.update(b)
                        #print("Key1 value=",h.hexdigest())
                        enc1=blake2b()
                        enc1.update(b'h')
                        #print("Key encryption1 value is=",enc1.hexdigest())
                
                        key1=0
                        for i in enc1.hexdigest():
                            key1=key1+ord(i)
                
                        key1=(map255(key1))
                        key2=keyval #Key2 is hardcoded here but through the web interface user can give value
                        dc, ac, blocks_count, tables=encoder()
                        dc, ac, blocks_count,tables=decoder(key1,dc, ac, blocks_count, tables,key2)
                        bitstream=[dc,ac,blocks_count,tables,randA,key1,key2]
                        
                        file=open("bQ","wb")
                        pickle.dump(bitstream,file)
                        file.close()
                        ms.showinfo('Success!','Encryption completed')
           
                        ###############Encryption body ends####################        
        root1.destroy()
        root = Tk()
        
        
        Label(root, width=50,height=15,bg='grey94',borderwidth=2,relief="groove").place(x=70,y=30)
        # Label(root, width=25,height=15,bg='lavender',borderwidth=2,relief="groove").place(x=240,y=30)
        
        image=Image.open("encryption.png")
        image=image.resize((50,50),Image.ANTIALIAS)
        img=ImageTk.PhotoImage(image)
        Label(root, image=img).place(x=230,y=40)
        
        
        
        
        image12=Image.open("key.png")
        image12=image12.resize((20,20),Image.ANTIALIAS)
        img12=ImageTk.PhotoImage(image12)
        Label(root, image=img12).place(x=99,y=134)
        
        key2 = Label(root, text="Encryption key:",fg='grey',bg='grey94',width=11,font=("bold", 10))
        key2.place(x=120,y=136)
        
        key22 = Entry(root,width=9,bg='thistle2')
        key22.place(x=215,y=140)
        
        
        
        
        image1=Image.open("find.png")
        image1=image1.resize((20,20),Image.ANTIALIAS)
        img1=ImageTk.PhotoImage(image1)
        Label(root, image=img1).place(x=97,y=178)
        
        label_0 = Label(root, text="Select image:",fg='grey',bg='grey94',width=9,font=("bold",10))
        label_0.place(x=120,y=180)
        
        Button(root, text='Browse',width=7,height=0,command=lambda:filechooser(root)).place(x=215,y=180)
        
        
        selected = Label(root, text="Selected image:",fg='grey',bg='grey94',width=16,font=("bold",8))
        selected.place_forget #(x=273,y=95)
        
        
        
       
        
        Button(root, text='Back',width=10,bg='SpringGreen4',fg='white',command=lambda:choose(root)).place(x=90,y=220)
        Button(root, text='Encrypt',width=15,bg='brown',fg='white',command=encrypt).place(x=170,y=220)
        window2=windows(root,"Encryption"," ")
                      


        
###################################################################################      
##########################Encryption part Ends####################################       
###################################################################################   

class windows:
    def __init__(self,root,title,message):
        self.root=root
        self.root.title("KSS-image encrypter")
        self.root.geometry('500x300')
        
        self.root.configure(bg='thistle2')
        
        self.root.iconbitmap("lcss1.ico")
       
        
        self.root.resizable(0, 0)
        
        
        self.root.update_idletasks()
        width = self.root.winfo_width()
        frm_width = self.root.winfo_rootx() - self.root.winfo_x()
        win_width = width + 2 * frm_width
        height = self.root.winfo_height()
        titlebar_height = self.root.winfo_rooty() - self.root.winfo_y()
        win_height = height + titlebar_height + frm_width
        x = self.root.winfo_screenwidth() // 2 - win_width // 2
        y = self.root.winfo_screenheight() // 2 - win_height // 2
        self.root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        self.root.deiconify()        
   
        
        
        
        self.root.mainloop()
        pass
    pass
main()