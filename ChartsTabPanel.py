import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ChartsTabPane:
    def __init__(self, root,figs):
        tabControl = ttk.Notebook(root)

        for figure in figs:
            tab = ttk.Frame(tabControl)
            tabControl.add(tab, text=figure.get_label())
            canvas = FigureCanvasTkAgg(figure,master=tab)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


        tabControl.pack(expand=1, fill="both")

        root.wait_window()


