from tkinter import *
from tkinter import Menu,ttk
tk =Tk()
canvas = Canvas(tk,width = 600,height = 500)
tk.title("Drawing circle")
canvas.grid()
coord_label = Label(tk, text="Координаты центра окружности: ",font=("Times New Roman", 16))
coord_label.place(x=0,y = 0)
radius_label = Label(tk,text="Радиус: ",font=("Times New Roman", 16))
radius_label.place(x=400,y =0)
x = str()
y = str()
r = str()
x_entry_label = Label(tk, text="x = ",font=("Times New Roman", 16))
x_entry_label.place(x=0, y=30)
x_entry = Entry()
x_entry.place(x=40, y=35)
y_entry_label = Label(tk, text="y = ",font=("Times New Roman", 16))
y_entry_label.place(x=200, y=30)
y_entry = Entry(tk, textvariable = y)
y_entry.place(x=240, y=35)
r_entry_label = Label(tk, text="r = ",font=("Times New Roman", 16))
r_entry_label.place(x=400, y=30)
r_entry = Entry(tk, textvariable = y)
r_entry.place(x=440, y=35)
x = float(x_entry.get())
y = int(y)
r = int(r)
btn_ok = ttk.Button(tk, text='Нарисовать круг')
btn_ok.place(x=0, y=60)
btn_ok.bind('<Button-1>', lambda event:canvas.create_oval(x-r, y-r, x+r, y+r))
tk.mainloop()