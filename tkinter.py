from tkinter.simpledialog import askstring
from tkinter import *


def input():
    text = askstring("Input", "please enter your message for spam detection")

button = Button(top, text = "Click")
button.place(x=500, y=100)

top.mainloop()