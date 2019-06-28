# coding utf-8

import os
import PIL
from tkinter import *

class Drawing(object):
    def __init__(self, width=200, height=200):
        self.width = width
        self.height = height
        self.center = self.height // 2
        self.white = (255, 255, 255)
        self.green = (0, 128, 0)
        self.root = Tk()
    
    def saveimg(self):
        self.file = os.path.join(self.filepath, self.filename.get()+".png")
        self.image.save(self.file)
        print("file saved {}".format(self.file))
        from time import sleep
        sleep(1)
        self.root.destroy()
    
    def paint(self, event):
        # python_green = "#476042"
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.cv.create_oval(x1, y1, x2, y2, fill="black",width=5)
        self.draw.line([x1, y1, x2, y2],fill="black",width=5)
        
    def main(self, preprocess=False, filepath="./"):
        from PIL import Image, ImageDraw
        import PIL
        self.filepath = filepath
        label = Label(self.root, text="Insert File Name", width=20, height=2, 
              anchor="w", justify="left")
        label.pack()
        self.filename = Entry(self.root)
        self.filename.pack()
        button=Button(self.root, text="Save&Quit", command=self.saveimg)
        button.pack()
        self.cv = Canvas(self.root, width=self.width, height=self.height, 
                    bg='white')
        self.cv.pack()
        self.image = PIL.Image.new("RGB", (self.width, self.height), 
                                   self.white)
        self.draw = ImageDraw.Draw(self.image)

        self.cv.pack(expand=YES, fill=BOTH)
        self.cv.bind("<B1-Motion>", self.paint)
        self.root.mainloop()
        
        if preprocess:
            
            img = Image.open(self.file)
            img = img.convert("L")
            img = PIL.ImageOps.invert(img)
            img.thumbnail((28, 28))
            img.save(self.file)


    