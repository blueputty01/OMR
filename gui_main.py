import json
from tkinter import *
from tkinter.filedialog import askopenfilename, askopenfilenames, asksaveasfilename
from tkinter.messagebox import showerror
import os

import spreadsheet_writer
import test_grader


class Application(Frame):
    def __init__(self):
        Frame.__init__(self)
        self.master.title("OMR")
        self.master.geometry("450x450")
        self.master.rowconfigure(5, weight=1)
        self.master.columnconfigure(5, weight=1)
        self.grid(sticky=W + E + N + S)

        self.output_browse = Button(self, text="Select output", command=self.set_export_loc)
        self.output_browse.grid(row=1, column=0, sticky=W)

        self.name_browse = Button(self, text="Open names", command=self.load_students)
        self.name_browse.grid(row=2, column=0, sticky=W)

    def load_key(self):
        file_path = askopenfilename(
            title="Select key",
            filetypes=(
                # ("Images", "*.png;*.jpg;*.jpeg;*.jpe;*.jp2;*webp;*pbm;*.bmp;*.dib"),
                ("All files", "*.*"),
                # "All files", "*.*"
            )
        )
        if not file_path:
            return
        try:
            test_grader.entry([file_path], False)
        except():  # <- naked except is a bad idea
            showerror("Read key", "Error occurred while reading image \n'%s'" % file_path)

        try:
            test_grader.write_key()
        except():  # <- naked except is a bad idea
            showerror("Read key", "Error occurred while writing key \n'%s'" % file_path)

        self.image_browse = Button(self, text="Open responses", command=self.load_responses)
        self.image_browse.grid(row=4, column=0, sticky=W)

    def load_responses(self):
        file_paths = askopenfilenames(
            title="Select responses",
            filetypes=(
                # ("Images", "*.png;*.jpg;*.jpeg;*.jpe;*.jp2;*webp;*pbm;*.bmp;*.dib"),
                ("All files", "*.*"),
                # "All files", "*.*"
            )
        )
        if not file_paths:
            return
        try:
            test_grader.entry(file_paths, True)
        except():  # <- naked except is a bad idea
            showerror("Open key image", "Failed to read file\n'%s'" % file_paths)
        print("Finished reading students. Now writing")
        try:
            test_grader.write_students()
        except():  # <- naked except is a bad idea
            showerror("Read key", "Error occurred while writing responses \n'%s'" % file_paths)
        print("Successfully wrote data")

    def load_students(self):
        file_path = askopenfilename(
            title="Select name file",
            filetypes=(
                ("JSON files", "*.json"),
            )
        )
        if not file_path:
            return
        f = open(file_path, encoding="utf8")
        test_grader.set_students(json.load(f))

    def set_export_loc(self):
        file = asksaveasfilename(title="Select save location", filetypes=(("Spreadsheet", ".*xlsx"),), defaultextension="xlsx")
        spreadsheet_writer.set_output(file)
        self.key_browse = Button(self, text="Open key", command=self.load_key)
        self.key_browse.grid(row=3, column=0, sticky=W)


if __name__ == "__main__":
    Application().mainloop()
