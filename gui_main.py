from tkinter import *
from tkinter.filedialog import askopenfilename, askopenfilenames
from tkinter.messagebox import showerror

from test_grader import entry

class Application(Frame):
    def __init__(self):
        Frame.__init__(self)
        self.master.title("OMR")
        self.master.geometry("450x450")
        self.master.rowconfigure(5, weight=1)
        self.master.columnconfigure(5, weight=1)
        self.grid(sticky=W + E + N + S)

        self.key_browse = Button(self, text="Browse for key", command=self.load_key)
        self.key_browse.grid(row=1, column=0, sticky=W)

    def load_key(self):
        file_path = askopenfilename(
            title="Select key",
            filetypes=(
                # ("Images", "*.png;*.jpg;*.jpeg;*.jpe;*.jp2;*webp;*pbm;*.bmp;*.dib"),
                ("All files", "*.*"),
                # "All files", "*.*"
            )
        )
        try:
            entry([file_path], False)
        except:  # <- naked except is a bad idea
            showerror("Open key image", "Failed to read file\n'%s'" % file_path)

        self.image_browse = Button(self, text="Browse for responses", command=self.load_responses)
        self.image_browse.grid(row=2, column=0, sticky=W)


    def load_responses(self):
        file_paths = askopenfilenames(
            title="Select responses",
            filetypes=(
                # ("Images", "*.png;*.jpg;*.jpeg;*.jpe;*.jp2;*webp;*pbm;*.bmp;*.dib"),
                ("All files", "*.*"),
                # "All files", "*.*"
            )
        )
        try:
            entry(file_paths, True)
        except:  # <- naked except is a bad idea
            showerror("Open key image", "Failed to read file\n'%s'" % file_paths)


if __name__ == "__main__":
    Application().mainloop()
