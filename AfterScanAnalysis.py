#!/usr/bin/env python
"""
AfterScanPlotChart - Analysis of AfterScan logs to help determine quality

Licensed under a MIT LICENSE.

More info in README.md file
"""

__author__ = 'Juan Remirez de Esparza'
__copyright__ = "Copyright 2022, Juan Remirez de Esparza"
__credits__ = ["Juan Remirez de Esparza"]
__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Juan Remirez de Esparza"
__email__ = "jremirez@hotmail.com"
__status__ = "Development"


import matplotlib.pyplot as plt
import csv
import sys
from tkinter import *
from tkinter import filedialog
import os
from datetime import datetime
import json
from json.decoder import JSONDecodeError

# Global variables
global win
general_config = {
}


# Configuration & support file vars
script_dir = os.path.realpath(sys.argv[0])
script_dir = os.path.dirname(script_dir)
general_config_filename = os.path.join(script_dir, "AfterScanAnalysis.json")


def display_plot(filename):
    x = []
    y = []

    with open(filename, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')

        for row in plots:
            x.append(int(row[3]))
            y.append(int(row[4]))
        for i in range(1, x[-1]):
            if i not in x:
                x.append(i)
                y.append(0)

    plt.bar(x, y, color='g', width=1, label="Missing rows")
    plt.xlabel('Frame')
    plt.ylabel('Missing')
    plt.title('Missing pixel rows')
    plt.legend()
    plt.show()


def show_text(message):
    global text_box
    text_box.config(state='normal')
    text_box.insert('end', message + '\n')
    text_box.config(state='disabled')


def select_log_file():
    project_list = []
    # show an "Open" dialog box and return the path to the selected file
    log_filename = filedialog.askopenfilename(filetypes=[("Log files", "*.log")])
    if not log_filename:
        return
    general_config["CurrentDir"] = os.path.split(log_filename)[0]
    # Determine how many projects inside
    for line in open(log_filename):
        if 'FrameAlignTag' in line:
            chunks = line.split(',')
            project = chunks[2]
            if project not in project_list:
                project_list.append(project)
    # Generate csvs from log file
    for project in project_list:
        csv_filename = temp = os.path.dirname(log_filename) + '/' + project + '.csv'
        # open(csv_filename,'w').writelines(line for line in open(log_filename) if 'FrameAlignTag' in line and project in line)
        faulty_frames = 0
        with open(csv_filename, 'w') as csv_file, open(log_filename) as log_file:
            for line in log_file:
                if 'FrameAlignTag' in line and project in line:
                    csv_file.writelines(line)
                    faulty_frames += 1
            show_text(project + ': %i frames out of bounds'%faulty_frames )


def select_csv_file():
    # show an "Open" dialog box and return the path to the selected file
    filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not filename:
        return
    general_config["CurrentDir"] = os.path.split(filename)[0]
    display_plot(filename)


def save_general_config():
    global general_config
    # Write config data upon exit
    general_config["GeneralConfigDate"] = str(datetime.now())
    with open(general_config_filename, 'w+') as f:
        json.dump(general_config, f)


def load_general_config():
    global general_config
    # Check if persisted data file exist: If it does, load it
    if os.path.isfile(general_config_filename):
        try:
            persisted_data_file = open(general_config_filename)
            general_config = json.load(persisted_data_file)
            persisted_data_file.close()
        except JSONDecodeError:
            pass
    else:   # No project config file. Set empty config to force defaults
        general_config = {}

    if 'CurrentDir' in general_config:
        if general_config["CurrentDir"] != '':
            os.chdir(general_config["CurrentDir"])


def exit_app():
    save_general_config()
    win.destroy()


def build_ui():
    global win
    global text_box
    main_frame = Frame(win)
    main_frame.pack(side=TOP, padx=5, pady=5)
    # Select log file button
    select_log_file_btn = Button(main_frame, text='Select Log File', width=20,
                               height=1, command=select_log_file,
                               activebackground='green',
                               activeforeground='white', padx=5, pady=5)
    select_log_file_btn.pack(side=TOP)
    # Select csv file button
    select_csv_file_btn = Button(main_frame, text='Select CSV File', width=20,
                               height=1, command=select_csv_file,
                               activebackground='green',
                               activeforeground='white', padx=5, pady=5)
    select_csv_file_btn.pack(side=TOP)
    # Application Exit button
    Exit_btn = Button(main_frame, text="Exit", width=20,
                      height=1, command=exit_app, activebackground='red',
                      activeforeground='white', padx=5, pady=5)
    Exit_btn.pack(side=TOP)
    text_box = Text(main_frame, height=20, width=70)
    text_box.pack(side=TOP)
    win.update_idletasks()

def app_init():
    global win

    win = Tk()  # Create main window, store it in 'win'

    load_general_config()

    app_width = 600
    app_height = 400

    win.title('AfterScan ' + __version__)  # setting title of the window
    win.geometry('600x400')  # setting the size of the window
    win.geometry('+50+50')  # setting the position of the window
    # Prevent window resize
    win.minsize(app_width, app_height)
    win.maxsize(app_width, app_height)
    win.update_idletasks()
    # Set default font size
    # Change the default Font that will affect in all the widgets
    win.option_add("*font", "TkDefaultFont 10")
    win.resizable(False, False)
    win.update_idletasks()


def main(argv):
    global win
    app_init()
    build_ui()
    #display_plot('/media/juan/Data 2/Video/Video edicion/ALT-Scann 8/Capturas/stabilize-reel 19&20.csv')
    # Main Loop
    win.mainloop()  # running the loop that works as a trigger

if __name__ == '__main__':
    main(sys.argv[1:])
