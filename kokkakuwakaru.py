import tkinter as tk
from tkinter import font
import tkinterdnd2 as tkdnd
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout
from keras.datasets import mnist
from keras.utils import np_utils
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import cv2
import os




def drop_enter(event):
    event.widget.focus_force()
    return event.action

def drop_leave(event):
    event.widget._root().focus_force()
    return event.action

def drop_position(event):
    #print(event.x_root, event.y_root)
    return event.action

def drop(event):
    if event.data:
        hantei(str(event.data).replace("{","").replace("}",""))
    return event.action

def hantei(path):
    global img
    img_width = 320
    img_height = 240
    
    num_data = 1
    
    save_data_path =os.getcwd()

    labels = ["ナチュラル","ストレート","ウェーブ"]

    model = model_from_json(open(save_data_path+"\\ex_datamodel.json",'r').read())

    model.load_weights(save_data_path+"\\ex_dataweight.hdf5")

    img = load_img(path,target_size=(img_width,img_height),grayscale=True)
    img = img_to_array(img)
    img = img.astype('float32')/255.0
    img = np.array([img])

    y_pred = model.predict(img)

    number_pred = np.argmax(y_pred)
    
    print("y_pred:",y_pred)
    print("number_bred:",number_pred)
    print("label_pred:",labels[int(number_pred)])
    canvas = tk.Canvas(bg="white", width=600, height=400)
    canvas.pack(expand = True, fill = tk.BOTH)
    if(number_pred==0):
        
        img = tk.PhotoImage(file=os.getcwd()+"\\natural.png")
 
        
        canvas.create_image(350,250,image=img )
    elif(number_pred==1):
        
        img = tk.PhotoImage(file=os.getcwd()+"\\strate.png")
 
        canvas.create_image(350,250,image=img )
    elif(number_pred==2):
        
        img = tk.PhotoImage(file=os.getcwd()+"\\wave.png")
 
        canvas.create_image(350,250,image=img)

root = tkdnd.Tk()
root.geometry("700x500")
root.title(u"Software Title")

font1 = font.Font(family='Helvetica', size=20, weight='bold')
label2 = tk.Label(root, text="骨格わかるちゃん", fg="white", bg="black", font=font1)
label2.pack(side="top")

font2 = font.Font(family='Helvetica', size=15, weight='bold')
label3 = tk.Label(root, text="画像をドラックアンドドロップ", fg="white", bg="black", font=font2)
label3.pack(side="top")

root.drop_target_register(tkdnd.DND_FILES)
root.dnd_bind('<<DropEnter>>', drop_enter)
root.dnd_bind('<<DropLeave>>', drop_leave)
root.dnd_bind('<<DropPosition>>', drop_position)
root.dnd_bind('<<Drop>>', drop)

root.mainloop() 
