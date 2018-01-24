import os
import Tkinter as tk
import tkMessageBox


class MyDialog:
    def __init__(self, parent, folder, person):
        self.folder = folder
        self.person = person

        top = self.top = tk.Toplevel(parent)
        w = top.winfo_reqwidth()
        h = top.winfo_reqheight()
        ws = top.winfo_screenwidth()
        hs = top.winfo_screenheight()
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        top.geometry('+%d+%d' % (x, y))
        top.title('Change client name')
        #top.wm_iconbitmap("@/home/maarten/Pictures/icon2.xbm")
        top.grid_rowconfigure(2, weight=1)
        top.grid_columnconfigure(2, weight=1)
        self.myLabel = tk.Label(top, text='New name: ')
        self.myLabel.grid(row=0, column=0, padx=10, pady=5, sticky='nsew')
        self.myEntryBox = tk.Entry(top)
        self.myEntryBox.grid(row=0, column=1, padx=10, pady=5, sticky='nsew')
        self.ok_button = tk.Button(top, text='OK', command=self.safe)
        self.ok_button.grid(row=1, column=0, padx=10, pady=5, sticky='nsew')
        self.cancel_button = tk.Button(top, text='Cancel', command=self.cancel)
        self.cancel_button.grid(row=1, column=1, padx=10, pady=5, sticky='nsew')

    def cancel(self):
        self.top.destroy()

    def safe(self):
        new_name = self.myEntryBox.get()

        if new_name != "":
            result = tkMessageBox.askquestion('Change name?',
                                              'Are you sure you want to change the name of "'
                                              + self.person + '" to "' + new_name + '"?',
                                              icon='warning')
            if result == "yes":
                new_name = new_name.replace(' ', '_')
                count = 1
                path = os.path.join(self.folder, self.person.replace(' ', '_'))
                for img in os.listdir(path):
                    if img.endswith('.png'):
                        img_new_name = new_name.replace(' ', '_') + '_' + str('%0*d' % (4, count)) + '.png'
                        os.rename(os.path.join(path, img), os.path.join(path, img_new_name))
                        count += 1
                os.rename(path, os.path.join(self.folder, new_name.replace(' ', '_')))
            self.top.destroy()
        else:
            tkMessageBox.showerror('Error!', 'Please enter a name.')