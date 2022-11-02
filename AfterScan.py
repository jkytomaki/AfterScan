#!/usr/bin/env python
"""
AfterScan - Basic post-processing for scanned R8/S8 films

This utility is intended to handle the basic post-processing after film
scanning is completed.

Actions performed by this tool include:
- Stabilization
- Cropping
- Video generation

Licensed under a MIT LICENSE.

More info in README.md file
"""

__author__ = 'Juan Remirez de Esparza'
__copyright__ = "Copyright 2022, Juan Remirez de Esparza"
__credits__ = ["Juan Remirez de Esparza"]
__license__ = "MIT"
__version__ = "1.5"
__maintainer__ = "Juan Remirez de Esparza"
__email__ = "jremirez@hotmail.com"
__status__ = "Development"

import tkinter as tk
from tkinter import filedialog

import tkinter.messagebox
from tkinter import *

import tkinter.font as tkfont

from PIL import ImageTk, Image

import os
import time
import subprocess as sp
import json
from datetime import datetime
import logging
import sys
import getopt
import cv2
import numpy as np
from glob import glob
import platform
import threading
import re
from enum import Enum

# Frame vars
first_absolute_frame = 0
last_absolute_frame = 0
frame_scale_refresh_done = True
frame_scale_refresh_pending = False
frames_to_encode = 0
CurrentFrame = 0
StartFrame = 0
global work_image, base_image, original_image

# Configuration & support file vars
script_dir = os.path.realpath(sys.argv[0])
script_dir = os.path.dirname(script_dir)
general_config_filename = os.path.join(script_dir, "AfterScan.json")
project_settings_filename = os.path.join(script_dir, "AfterScan-projects.json")
project_config_basename = "AfterScan-project.json"
project_config_filename = ""
project_config_from_file = True
job_list_filename = os.path.join(script_dir, "AfterScan_job_list.json")
pattern_filename_r8 = os.path.join(script_dir, "Pattern.R8.jpg")
pattern_filename_s8 = os.path.join(script_dir, "Pattern.S8.jpg")
pattern_filename = pattern_filename_s8

general_config = {
}
default_project_config = {
    'FrameInputFilenamePattern': 'picture-*.jpg',
    'FilmType': 'S8'
}
project_config = default_project_config.copy()


# Film hole search vars
expected_pattern_pos_s8 = (6.5, 34)
expected_pattern_pos_r8 = (6.8, 15.5)   # used bo be 9.6, 13.3 before shortening height of R8 template
expected_pattern_pos = expected_pattern_pos_s8
default_hole_height_s8 = 344
default_interhole_height_r8 = 808
pattern_bw_filename = os.path.join(script_dir, "Pattern_BW.jpg")
pattern_wb_filename = os.path.join(script_dir, "Pattern_WB.jpg")
film_hole_height = 0
film_hole_template = None
HoleSearchTopLeft = (0, 0)
HoleSearchBottomRight = (0, 0)

# Film frames (in/out) file vars
TargetVideoFilename = ""
SourceDir = ""
TargetDir = ""
FrameFilenameOutputPattern = "picture_out-%05d.jpg"
FrameCheckFilenameOutputPattern = "picture_out-*.jpg"  # Req. for ffmpeg gen.
SourceDirFileList = []
global film_type

# Flow control vars
ConvertLoopExitRequested = False
ConvertLoopRunning = False
BatchJobRunning = False

# preview dimensions (4/3 format) vars
PreviewWidth = 700
PreviewHeight = 525
PreviewRatio = 1  # Defined globally for homogeneity, to be calculated once per project
SmallSize = False

# Crop area rectangle drawing vars
ref_point = []
rectangle_drawing = False  # true if mouse is pressed
ix, iy = -1, -1
x_, y_ = 0, 0
CropWindowTitle = "Select area to crop, press Enter to confirm, " \
                  "Escape to cancel"
StabilizeWindowTitle = "Select height of S8 hole, or R8 inter-hole space. " \
                       "Press Enter to confirm, Escape to cancel"
RectangleWindowTitle = ""
StabilizeAreaDefined = False
StabilizationThreshold = 240.0
CropAreaDefined = False
RectangleTopLeft = (0, 0)
RectangleBottomRight = (0, 0)
CropTopLeft = (0, 0)
CropBottomRight = (0, 0)

# Video generation vars
VideoFps = 18
FfmpegBinName = ""
ui_init_done = False
IgnoreConfig = False
global ffmpeg_installed
ffmpeg_state = Enum('ffmpeg_state', ['Pending', 'Running', 'Completed'])
resolution_dict = {
    "Unchanged": "",
    "-- 4:3 --": "",
    "160x120 (QQVGA)": "160:120",
    "320x240 (QVGA)": "320:240",
    "640x480 (VGA)": "640:480",
    "800x600 (SVGA)": "800:600",
    "1024x768 (XGA)": "1024:768",
    "1152x864 (XGA+)": "1152:864",
    "1280x960 (SXGA−)": "1280:960",
    "1400x1050 (SXGA+)": "1400:1050",
    "1600x1200 (UXGA)": "1600:1200",
    "1920x1440 (1080P)": "1920:1440",
    "2048x1536 (QXGA)": "2048:1536",
    "2880x2160 (3K UHD)": "2880:2160",
    "3072x2304 (3K)": "3072:2304",
    "3840x2880 (4K UHD)": "3840:2880",
    "4096x3072 (HXGA)": "4096:3072",
    "5120x3840 (5K)": "5120:3840",
    "6144x4608 (6K)": "6144:4608",
    "7680x5760 (8K UHD)": "7680:5760",
    "8192x6144 (8K)": "8192:6144",
    "-- 16:9 --": "",
    "432x243 (FWQVGA)": "432:243",
    "640x360 (nHD)": "640:360",
    "896x504 (FWVGA)": "896:504",
    "960x540 (qHD)": "960:540",
    "1024x576 (EDTV)": "1024:576",
    "1280x720 (HD Ready)": "1280:720",
    "1360x765 (WXGA)": "1360:765",
    "1600x900 (HD+)": "1600:900",
    "1920x1080 (FHD)": "1920:1080",
    "2048x1152 (2K)": "2048:1152",
    "2560x1440 (QHD)": "2560:1440",
    "3072x1728 (3K)": "3072:1728",
    "3200x1800 (QHD+)": "3200:1800",
    "3840x2160 (4K-UHD)": "3840:2160",
    "4096x2304 (DCI 4K)": "4096:2304",
    "5120x2880 (5K UHD+)": "5120:2880",
    "7680x4320 (8K-UHD)": "7680:4320",
    "8192x4608 (True 8K)": "8192:4608",
    "15360x8640 (16K UHD)": "15360:8640"
}
# Miscellaneous vars
global win
ExpertMode = False
IsWindows = False
IsLinux = False
IsMac = False

"""
#################
Utility functions
#################
"""


# Define a function for
# identifying a Digit
def is_a_number(string):
    # Make a regular expression
    # for identifying a digit
    regex = '^[0-9]+$'
    # pass the regular expression
    # and the string in search() method
    if (re.search(regex, string)):
        return True
    else:
        return False
"""
####################################
Configuration file support functions
####################################
"""


def save_general_config():
    # Write config data upon exit
    general_config["GeneralConfigDate"] = str(datetime.now())
    if not IgnoreConfig:
        with open(general_config_filename, 'w+') as f:
            json.dump(general_config, f)


def load_general_config():
    global general_config
    global general_config_filename
    global LastSessionDate
    global SourceDir, TargetDir

    # Check if persisted data file exist: If it does, load it
    if not IgnoreConfig and os.path.isfile(general_config_filename):
        persisted_data_file = open(general_config_filename)
        general_config = json.load(persisted_data_file)
        persisted_data_file.close()
    else:   # No project config file. Set empty config to force defaults
        general_config = {}

    for item in general_config:
        logging.info("%s=%s", item, str(general_config[item]))

    if 'SourceDir' in general_config:
        SourceDir = general_config["SourceDir"]
        # If directory in configuration does not exist, set current working dir
        if not os.path.isdir(SourceDir):
            SourceDir = ""
    if 'FfmpegBinName' in general_config:
        FfmpegBinName = general_config["FfmpegBinName"]


def update_project_settings():
    global project_settings
    global SourceDir
    # SourceDir is the key for each project config inside the global project settings
    if SourceDir in project_settings:
        project_settings.update({SourceDir: project_config.copy()})
    elif SourceDir != '':
        project_settings[project_config["SourceDir"]] = project_config.copy()


def save_project_settings():
    global project_settings, project_settings_filename

    if not IgnoreConfig:
        with open(project_settings_filename, 'w+') as f:
            logging.debug(project_settings)
            json.dump(project_settings, f)


def load_project_settings():
    global project_settings, project_settings_filename

    if not IgnoreConfig and os.path.isfile(project_settings_filename):
        f = open(project_settings_filename)
        project_settings = json.load(f)
        f.close()
    else:   # No project settings file. Set empty config to force defaults
        project_settings = {}


def save_project_config():
    global skip_frame_regeneration
    global frames_to_encode_spinbox
    global ffmpeg_preset
    global StabilizeAreaDefined, film_hole_height
    global CurrentFrame
    global video_filename_name

    # Do not save if current project comes from batch job
    if not project_config_from_file or IgnoreConfig:
        return
    # Write project data upon exit
    project_config["SourceDir"] = SourceDir
    project_config["CurrentFrame"] = CurrentFrame
    project_config["FramesToEncode"] = frames_to_encode_spinbox.get()
    project_config["skip_frame_regeneration"] = skip_frame_regeneration.get()
    project_config["FFmpegPreset"] = ffmpeg_preset.get()
    project_config["ProjectConfigDate"] = str(datetime.now())
    project_config["PerformCropping"] = perform_cropping.get()
    project_config["VideoFilename"] = video_filename_name.get()
    if StabilizeAreaDefined:
        project_config["HoleHeight"] = film_hole_height
        project_config["PerformStabilization"] = perform_stabilization.get()
    if ExpertMode:
        project_config["FillBorders"] = fill_borders.get()
        project_config["FillBordersThickness"] = fill_borders_thickness.get()
        project_config["FillBordersMode"] = fill_borders_mode.get()

    # No longer saving to dedicated file, all project settings in common file now
    # with open(project_config_filename, 'w+') as f:
    #     json.dump(project_config, f)

    update_project_settings()
    save_project_settings()

def load_project_config():
    global SourceDir
    global project_config, project_config_from_file
    global project_config_basename, project_config_filename
    global project_settings

    project_config_filename = os.path.join(SourceDir, project_config_basename)
    # Check if persisted project data file exist: If it does, load it
    project_config = default_project_config.copy()  # set default config
    if not IgnoreConfig:
        if SourceDir in project_settings:
            logging.debug("Loading project config from consolidated project settings")
            project_config |= project_settings[SourceDir].copy()
        elif os.path.isfile(project_config_filename):
            logging.debug("Loading project config from dedicated project config file")
            persisted_data_file = open(project_config_filename)
            project_config |= json.load(persisted_data_file)
            persisted_data_file.close()
        else:  # No project config file. Set empty config to force defaults
            project_config = default_project_config.copy()

    for item in project_config:
        logging.info("%s=%s", item, str(project_config[item]))

    # Allow to determine source of current project, to avoid
    # saving it in case of batch processing
    project_config_from_file = True


def decode_project_config():        
    global SourceDir, TargetDir
    global project_config
    global project_config_basename, project_config_filename
    global CurrentFrame, frame_slider
    global VideoFps, video_fps_dropdown_selected
    global resolution_dropdown, resolution_dropdown_selected
    global frame_input_filename_pattern
    global start_from_current_frame, frames_to_encode
    global skip_frame_regeneration
    global generate_video, video_filename_name
    global CropTopLeft, CropBottomRight, perform_cropping
    global StabilizeAreaDefined, film_hole_height, film_type
    global ExpertMode
    global StabilizationThreshold

    if 'SourceDir' in project_config:
        SourceDir = project_config["SourceDir"]
        # If directory in configuration does not exist, set current working dir
        if not os.path.isdir(SourceDir):
            SourceDir = ""
        folder_frame_source_dir.delete(0, 'end')
        folder_frame_source_dir.insert('end', SourceDir)
    if 'TargetDir' in project_config:
        TargetDir = project_config["TargetDir"]
        # If directory in configuration does not exist, set current working dir
        if not os.path.isdir(TargetDir):
            TargetDir = ""
        folder_frame_target_dir.delete(0, 'end')
        folder_frame_target_dir.insert('end', TargetDir)
    if 'CurrentFrame' in project_config:
        CurrentFrame = project_config["CurrentFrame"]
        CurrentFrame = max(CurrentFrame, 0)
        frame_slider.set(CurrentFrame)
    else:
        CurrentFrame = 0
        frame_slider.set(CurrentFrame)
    if 'StartFromCurrentFrame' in project_config:
        start_from_current_frame.set(project_config["StartFromCurrentFrame"])
    else:
        start_from_current_frame.set(False)
    if 'FramesToEncode' in project_config:
        frames_to_encode_str.set(project_config["FramesToEncode"])
        if frames_to_encode_str.get() == 'All':
            frames_to_encode = 0
        else:
            frames_to_encode = int(frames_to_encode_str.get())
        # frames_to_encode_spinbox.set(frames_to_encode)
    else:
        frames_to_encode = "All"
        frames_to_encode_str.set(frames_to_encode)

    if not 'FilmType' in project_config:
        project_config["FilmType"] = 'S8'
    film_type.set(project_config["FilmType"])
    set_film_type()

    if 'StabilizationThreshold' in project_config:
        StabilizationThreshold = project_config["StabilizationThreshold"]
        stabilization_threshold_str.set(StabilizationThreshold)
    else:
        StabilizationThreshold = 240
        stabilization_threshold_str.set(StabilizationThreshold)
    if 'PerformCropping' in project_config:
        perform_cropping.set(project_config["PerformCropping"])
    else:
        perform_cropping.set(False)
    if 'CropRectangle' in project_config:
        CropBottomRight = tuple(project_config["CropRectangle"][1])
        CropTopLeft = tuple(project_config["CropRectangle"][0])
    else:
        CropBottomRight = (0, 0)
        CropTopLeft = (0, 0)
    perform_cropping_selection()
    if 'GenerateVideo' in project_config:
        generate_video.set(project_config["GenerateVideo"])
    else:
        generate_video.set(False)
    generate_video_selection()
    if 'VideoFilename' in project_config:
        TargetVideoFilename = project_config["VideoFilename"]
        video_filename_name.delete(0, 'end')
        video_filename_name.insert('end', TargetVideoFilename)
    else:
        video_filename_name.delete(0, 'end')
    if 'skip_frame_regeneration' in project_config:
        skip_frame_regeneration.set(project_config["skip_frame_regeneration"])
    else:
        skip_frame_regeneration.set(False)
    if not 'FrameInputFilenamePattern' in project_config:
        project_config["FrameInputFilenamePattern"] = "picture-*.jpg"
    frame_input_filename_pattern.delete(0, 'end')
    frame_input_filename_pattern.insert('end', project_config["FrameInputFilenamePattern"])
    if 'FFmpegPreset' in project_config:
        ffmpeg_preset.set(project_config["FFmpegPreset"])
    else:
        ffmpeg_preset.set("veryfast")

    if 'HoleHeight' in project_config:
        film_hole_height = project_config["HoleHeight"]
        StabilizeAreaDefined = True
        perform_stabilization_checkbox.config(state=NORMAL)
    else:
        film_hole_height = 0
        StabilizeAreaDefined = False
        perform_stabilization_checkbox.config(state=DISABLED)

    if 'PerformStabilization' in project_config:
        perform_stabilization.set(project_config["PerformStabilization"])
    else:
        perform_stabilization.set(False)

    if 'VideoFps' in project_config:
        VideoFps = eval(project_config["VideoFps"])
        video_fps_dropdown_selected.set(VideoFps)
    else:
        VideoFps = 18
        video_fps_dropdown_selected.set(VideoFps)
    set_fps(str(VideoFps))
    if 'VideoResolution' in project_config:
        resolution_dropdown_selected.set(project_config["VideoResolution"])
    else:
        resolution_dropdown_selected.set('Unchanged')


    if ExpertMode:
        if 'FillBorders' in project_config:
            fill_borders.set(project_config["FillBorders"])
        else:
            fill_borders.set(False)
        if 'FillBordersThickness' in project_config:
            fill_borders_thickness.set(project_config["FillBordersThickness"])
        else:
            fill_borders_thickness.set(5)

        if 'FillBordersMode' in project_config:
            fill_borders_mode.set(project_config["FillBordersMode"])
        else:
            fill_borders_mode.set('smear')

    widget_state_refresh()

    win.update()


"""
##########################
Job list support functions
##########################
"""
def job_list_add_current():
    global job_list
    global CurrentFrame, frames_to_encode
    global project_config, video_filename_name
    global job_list_listbox
    global start_from_current_frame, SourceDirFileList

    entry_name = video_filename_name.get()
    if entry_name == "":
        entry_name = os.path.split(SourceDir)[1]
    if project_config["FilmType"] == 'R8':
        entry_name = entry_name + ", R8, "
    else:
        entry_name = entry_name + ", S8, "
    if frames_to_encode > 0:
        entry_name = entry_name + str(frames_to_encode)
    else:
        entry_name = entry_name + str(len(SourceDirFileList))
    entry_name = entry_name + " frames from frame "
    if start_from_current_frame.get():
        entry_name = entry_name + str(CurrentFrame)
    else:
        entry_name = entry_name + "0"
    if project_config["GenerateVideo"]:
        if ffmpeg_preset.get() == 'veryslow':
            entry_name = entry_name + ", HQ video"
        elif ffmpeg_preset.get() == 'veryfast':
            entry_name = entry_name + ", Low Q. video"
        else:
            entry_name = entry_name + ", medium Q. video"
    else:
        entry_name = entry_name + ", no video"
    if entry_name in job_list:
        tk.messagebox.showerror(
            "Error: Job already exists",
            "A job named " + entry_name + " exists already in the job list. "
            "Please delete existing job or rename this one before retrying.")
    else:
        save_project_config()  # Make sure all current settings are in project_config
        job_list[entry_name] = {'project': project_config.copy(), 'done': False}
        job_list_listbox.insert('end', entry_name)


def job_list_delete_selected():
    global job_list
    global job_list_listbox
    selected = job_list_listbox.curselection()
    job_list.pop(job_list_listbox.get(selected))
    if selected != ():
        job_list_listbox.delete(selected)


def save_job_list():
    global job_list, job_list_filename

    if not IgnoreConfig:
        with open(job_list_filename, 'w+') as f:
            json.dump(job_list, f)


def load_job_list():
    global job_list, job_list_filename

    if not IgnoreConfig and os.path.isfile(job_list_filename):
        f = open(job_list_filename)
        job_list = json.load(f)
        for entry in job_list:
            job_list_listbox.insert('end', entry)
        f.close()
    else:   # No job list file. Set empty config to force defaults
        job_list = {}



def start_processing_job_list():
    global BatchJobRunning, start_batch_btn
    BatchJobRunning = True
    button_status_change_except(start_batch_btn, DISABLED)
    job_processing_loop()


def job_processing_loop():
    global job_list
    global project_config
    global CurrentJobEntry
    global BatchJobRunning
    global project_config_from_file

    job_started = False
    idx = 0
    for entry in job_list:
        if  job_list[entry]['done'] == False:
            job_list_listbox.selection_set(idx)
            CurrentJobEntry = entry
            logging.info("Processing %s, starting from frame %i, %s frames",
                         entry, job_list[entry]['project']['CurrentFrame'],
                         job_list[entry]['project']['FramesToEncode'])
            project_config_from_file = False
            project_config = job_list[entry]['project'].copy()
            decode_project_config()

            # Load matching file list from newly selected dir
            get_current_dir_file_list()  # first_absolute_frame is set here

            start_convert()
            job_started = True
            break
        job_list_listbox.selection_clear(idx)
        idx += 1
    if not job_started:
        BatchJobRunning = False
        generation_exit()


"""
###############################
User feedback support functions
###############################
"""


def display_ffmpeg_progress():
    global win
    global ffmpeg_process
    global stop_event, stop_event_lock
    global TargetDir, TargetVideoFilename

    exit_thread = False
    while not exit_thread:
        time.sleep(0.5)
        if stop_event.is_set():
            exit_thread = True
        while True:
            line = ffmpeg_process.stdout.readline()
            if not line:
                break
            else:
                # Now this is part of the tool functionality, so we display
                # it using print, to avoid dependencies on log level
                print(time.strftime("%H:%M:%S", time.localtime()) + " - " +
                      str(line)[:-1])
                encoded_frame = str(line)[:-1].split()[1]


def display_ffmpeg_result(ffmpeg_output):
    global win

    ffmpeg_result = Toplevel(win)
    ffmpeg_result.title('Video encoding completed. Results displayed below')
    ffmpeg_result.geometry('600x400')
    ffmpeg_result.geometry('+250+250')
    ffmpeg_label = Text(ffmpeg_result, borderwidth=0)
    ffmpeg_label.insert(1.0, ffmpeg_output)
    ffmpeg_label.pack(side=TOP)
    # creating and placing scrollbar
    ffmpeg_result_sb = Scrollbar(ffmpeg_result, orient=VERTICAL)
    ffmpeg_result_sb.pack(side=RIGHT)
    # binding scrollbar with other widget (Text, Listbox, Frame, etc)
    ffmpeg_label.config(yscrollcommand=ffmpeg_result_sb.set)
    ffmpeg_result_sb.config(command=ffmpeg_label.yview)


"""
#######################
File handling functions
#######################
"""


def set_source_folder():
    global SourceDir, CurrentFrame, frame_slider, Go_btn, cropping_btn
    global first_absolute_frame

    # Write project data before switching project
    save_project_config()

    SourceDir = tk.filedialog.askdirectory(
        initialdir=SourceDir,
        title="Select folder with captured images to process")

    if not SourceDir:
        return
    elif TargetDir == SourceDir:
        tk.messagebox.showerror(
            "Error!",
            "Source folder cannot be the same as target folder.")
        return
    else:
        folder_frame_source_dir.delete(0, 'end')
        folder_frame_source_dir.insert('end', SourceDir)

    general_config["SourceDir"] = SourceDir

    load_project_config()  # Needs SourceDir defined

    # Load matching file list from newly selected dir
    get_current_dir_file_list()  # first_absolute_frame is set here

    decode_project_config()  # Needs first_absolute_frame defined

    # Enable Start and Crop buttons, plus slider, once we have files to handle
    cropping_btn.config(state=NORMAL)
    frame_slider.config(state=NORMAL)
    Go_btn.config(state=NORMAL)
    frame_slider.set(CurrentFrame)
    init_display()


def set_target_folder():
    global TargetDir
    global folder_frame_target_dir

    TargetDir = tk.filedialog.askdirectory(
        initialdir=TargetDir,
        title="Select folder where to store processed images/video")

    if not TargetDir:
        return
    elif TargetDir == SourceDir:
        tk.messagebox.showerror(
            "Error!",
            "Target folder cannot be the same as source folder.")
        return
    else:
        folder_frame_target_dir.delete(0, 'end')
        folder_frame_target_dir.insert('end', TargetDir)

    project_config["TargetDir"] = TargetDir


"""
###############################
UI support commands & functions
###############################
"""


def button_status_change_except(except_button, button_status):
    global source_folder_btn, target_folder_btn
    global perform_stabilization_checkbox
    global perform_cropping_checkbox, Crop_btn
    global Go_btn
    global Exit_btn

    if except_button != source_folder_btn:
        source_folder_btn.config(state=button_status)
    if except_button != target_folder_btn:
        target_folder_btn.config(state=button_status)
    if except_button != perform_cropping_checkbox:
        perform_cropping_checkbox.config(state=button_status)
    # if except_button != Crop_btn:
    #    Crop_btn.config(state=DISABLED if active else NORMAL)
    if except_button != Go_btn:
        Go_btn.config(state=button_status)
    if except_button != Exit_btn:
        Exit_btn.config(state=button_status)
    if except_button != perform_stabilization_checkbox:
        perform_stabilization_checkbox.config(state=button_status)

    if not CropAreaDefined:
        perform_cropping_checkbox.config(state=DISABLED)
    if ExpertMode:
        if except_button != start_batch_btn:
            start_batch_btn.config(state=button_status)
        if except_button != add_job_btn:
            add_job_btn.config(state=button_status)
        if except_button != delete_job_btn:
            delete_job_btn.config(state=button_status)


def widget_state_refresh():
    global perform_cropping_checkbox, perform_cropping, CropAreaDefined
    global CropTopLeft, CropBottomRight
    global ffmpeg_installed, perform_cropping
    global generate_video, generate_video_checkbox
    global fill_borders, fill_borders_checkbox
    global fill_borders_thickness_slider, fill_borders_mode_label
    global fill_borders_mode_label_dropdown
    global video_fps_dropdown, video_fps_label, video_filename_name
    global resolution_dropdown, resolution_label
    global ffmpeg_preset_rb1, ffmpeg_preset_rb2, ffmpeg_preset_rb3
    global ExpertMode

    if CropTopLeft != (0, 0) and CropBottomRight != (0, 0):
        CropAreaDefined = True
    perform_cropping_checkbox.config(
        state=NORMAL if CropAreaDefined else DISABLED)
    generate_video_checkbox.config(
        state=NORMAL if ffmpeg_installed else DISABLED)
    video_fps_dropdown.config(
        state=NORMAL if generate_video.get() else DISABLED)
    video_fps_label.config(
        state=NORMAL if generate_video.get() else DISABLED)
    resolution_dropdown.config(
        state=NORMAL if generate_video.get() else DISABLED)
    resolution_label.config(
        state=NORMAL if generate_video.get() else DISABLED)
    video_filename_name.config(
        state=NORMAL if generate_video.get() else DISABLED)
    ffmpeg_preset_rb1.config(
        state=NORMAL if generate_video.get() else DISABLED)
    ffmpeg_preset_rb2.config(
        state=NORMAL if generate_video.get() else DISABLED)
    ffmpeg_preset_rb3.config(
        state=NORMAL if generate_video.get() else DISABLED)
    if ExpertMode:
        fill_borders_checkbox.config(
            state=NORMAL if generate_video.get() else DISABLED)
        fill_borders_thickness_slider.config(
            state=NORMAL if generate_video.get() else DISABLED)
        fill_borders_mode_label.config(
            state=NORMAL if generate_video.get() else DISABLED)
        fill_borders_mode_label_dropdown.config(
            state=NORMAL if generate_video.get() else DISABLED)


def frame_input_filename_pattern_focus_out(event):
    global frame_input_filename_pattern

    project_config["FrameInputFilenamePattern"] = frame_input_filename_pattern.get()
    get_current_dir_file_list()
    init_display()


def custom_ffmpeg_path_focus_out(event):
    global custom_ffmpeg_path, FfmpegBinName

    if not is_ffmpeg_installed():
        tk.messagebox.showerror("Error!",
                                "Provided FFMpeg path is invalid.")
        custom_ffmpeg_path.delete(0, 'end')
        custom_ffmpeg_path.insert('end', FfmpegBinName)
    else:
        FfmpegBinName = custom_ffmpeg_path.get()
        general_config["FfmpegBinName"] = FfmpegBinName


def perform_stabilization_selection():
    global perform_stabilization
    stabilization_threshold_spinbox.config(
        state=NORMAL if perform_stabilization.get() else DISABLED)
    project_config["PerformStabilization"] = perform_stabilization.get()
        


def stabilization_threshold_selection(updown):
    global stabilization_threshold_spinbox, stabilization_threshold_str
    global StabilizationThreshold
    StabilizationThreshold = stabilization_threshold_spinbox.get()
    project_config["StabilizationThreshold"] = StabilizationThreshold


def stabilization_threshold_spinbox_focus_out(event):
    global stabilization_threshold_spinbox, stabilization_threshold_str
    global StabilizationThreshold
    StabilizationThreshold = stabilization_threshold_spinbox.get()
    project_config["StabilizationThreshold"] = StabilizationThreshold


def perform_cropping_selection():
    global perform_cropping, perform_cropping
    global generate_video_checkbox
    global ui_init_done
    generate_video_checkbox.config(state=NORMAL if ffmpeg_installed
                                   else DISABLED)
    project_config["PerformCropping"] = perform_cropping.get()
    if ui_init_done:
        scale_display_update()


def start_from_current_frame_selection():
    global start_from_current_frame
    project_config["StartFromCurrentFrame"] = start_from_current_frame.get()


def frames_to_encode_selection(updown):
    global frames_to_encode_spinbox, frames_to_encode_str, frames_to_encode
    project_config["FramesToEncode"] = frames_to_encode_spinbox.get()
    frames_to_encode = int(frames_to_encode_str.get())
    if project_config["FramesToEncode"] == '0':
        if updown == 'up':
            frames_to_encode_str.set('1')
        else:
            frames_to_encode_str.set('All')


def frames_to_encode_spinbox_focus_out(event):
    global frames_to_encode_spinbox, frames_to_encode_str, frames_to_encode
    project_config["FramesToEncode"] = frames_to_encode_spinbox.get()
    frames_to_encode = int(project_config["FramesToEncode"])


def fill_borders_selection():
    global fill_borders
    project_config["FillBorders"] = fill_borders.get()


def fill_borders_set_mode(selected):
    global fill_borders_mode

    fill_borders_mode.set(selected)
    project_config["FillBordersMode"] = fill_borders_mode.get()


def fill_borders_set_thickness_scale(selected_thickness):
    global fill_borders_thickness

    fill_borders_thickness_slider.focus()
    fill_borders_thickness.set(selected_thickness)
    project_config["FillBordersThinkness"] = fill_borders_thickness.get()


def generate_video_selection():
    global generate_video
    global video_fps_dropdown, video_fps_label, video_filename_name
    global ffmpeg_preset_rb1, ffmpeg_preset_rb2, ffmpeg_preset_rb3
    global resolution_dropdown, resolution_label

    project_config["GenerateVideo"] = generate_video.get()
    video_fps_dropdown.config(
        state=NORMAL if generate_video.get() else DISABLED)
    video_fps_label.config(
        state=NORMAL if generate_video.get() else DISABLED)
    resolution_dropdown.config(
        state=NORMAL if generate_video.get() else DISABLED)
    resolution_label.config(
        state=NORMAL if generate_video.get() else DISABLED)
    video_filename_name.config(
        state=NORMAL if generate_video.get() else DISABLED)
    ffmpeg_preset_rb1.config(
        state=NORMAL if generate_video.get() else DISABLED)
    ffmpeg_preset_rb2.config(
        state=NORMAL if generate_video.get() else DISABLED)
    ffmpeg_preset_rb3.config(
        state=NORMAL if generate_video.get() else DISABLED)
    if ExpertMode:
        fill_borders_checkbox.config(
            state=NORMAL if generate_video.get() else DISABLED)
        fill_borders_thickness_slider.config(
            state=NORMAL if generate_video.get() else DISABLED)
        fill_borders_mode_label.config(
            state=NORMAL if generate_video.get() else DISABLED)
        fill_borders_mode_label_dropdown.config(
            state=NORMAL if generate_video.get() else DISABLED)


def set_fps(selected):
    global VideoFps

    project_config["VideoFps"] = selected
    VideoFps = eval(selected)


def set_resolution(selected):
    global resolution_dict
    project_config["VideoResolution"] = selected


def scale_display_update():
    global win
    global frame_scale_refresh_done, frame_scale_refresh_pending
    global CurrentFrame
    global perform_stabilization, perform_cropping
    global CropTopLeft, CropBottomRight
    global SourceDirFileList

    if CurrentFrame >= len(SourceDirFileList):
        return
    file = SourceDirFileList[CurrentFrame]
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    if perform_stabilization.get():
        img = stabilize_image(img)
    if perform_cropping.get():
        img = crop_image(img, CropTopLeft, CropBottomRight)
    else:
        img = even_image(img)
    display_image(img)
    frame_scale_refresh_done = True
    if frame_scale_refresh_pending:
        frame_scale_refresh_pending = False
        win.after(100, scale_display_update)


def select_scale_frame(selected_frame):
    global win
    global SourceDir
    global CurrentFrame
    global SourceDirFileList
    global first_absolute_frame
    global frame_scale_refresh_done, frame_scale_refresh_pending
    global frame_slider

    if not ConvertLoopRunning:  # Do not refresh during conversion loop
        frame_slider.focus()
        CurrentFrame = int(selected_frame)
        project_config["CurrentFrame"] = CurrentFrame
        frame_slider.config(label='Global:'+
                            str(CurrentFrame+first_absolute_frame))
        if frame_scale_refresh_done:
            frame_scale_refresh_done = False
            frame_scale_refresh_pending = False
            win.after(5, scale_display_update)
        else:
            frame_scale_refresh_pending = True


"""
##############################
Second level support functions
##############################
(Code below to draw a rectangle to select area to crop or find hole,
adapted from various authors in Stack Overflow)
"""


def draw_rectangle(event, x, y, flags, param):
    global work_image, base_image, original_image
    global rectangle_drawing
    global ix, iy
    global x_, y_
    # Code posted by Ahsin Shabbir, same Stack overflow thread
    global RectangleTopLeft, RectangleBottomRight
    global area_select_image_factor

    if event == cv2.EVENT_LBUTTONDOWN:
        if not rectangle_drawing:
            work_image = np.copy(base_image)
            x_, y_ = -10, -10
            ix, iy = -10, -10
        rectangle_drawing = True
        ix, iy = x, y
        x_, y_ = x, y
    elif event == cv2.EVENT_MOUSEMOVE and rectangle_drawing:
        copy = work_image.copy()
        x_, y_ = x, y
        cv2.rectangle(copy, (ix, iy), (x_, y_), (0, 255, 0), 1)
        cv2.imshow(RectangleWindowTitle, copy)
    elif event == cv2.EVENT_LBUTTONUP:
        rectangle_drawing = False
        cv2.rectangle(work_image, (ix, iy), (x, y), (0, 255, 0), 1)
        # Update global variables with area
        # Need to account for the fact area calculated with 50% reduced image
        RectangleTopLeft = (max(0, round(min(ix, x)/area_select_image_factor)),
                            max(0, round(min(iy, y)/area_select_image_factor)))
        RectangleBottomRight = (min(original_image.shape[1], round(max(ix, x)/area_select_image_factor)),
                                min(original_image.shape[0], round(max(iy, y)/area_select_image_factor)))
        logging.debug("Original image: (%i, %i)", original_image.shape[1], original_image.shape[0])
        logging.debug("Selected area: (%i, %i), (%i, %i)",
                      RectangleTopLeft[0], RectangleTopLeft[1],
                      RectangleBottomRight[0], RectangleBottomRight[1])


def select_rectangle_area():
    global work_image, base_image, original_image
    global CurrentFrame, first_absolute_frame
    global SourceDirFileList
    global rectangle_drawing
    global ix, iy
    global x_, y_
    global area_select_image_factor

    if CurrentFrame >= len(SourceDirFileList):
        return False

    retvalue = False
    ix, iy = -1, -1
    x_, y_ = 0, 0

    file = SourceDirFileList[CurrentFrame]

    # load the image, clone it, and setup the mouse callback function
    original_image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    # Stabilize image to make sure target image matches user visual definition
    original_image = stabilize_image(original_image)
    # Scale area selection image as required
    work_image = resize_image(original_image, 100*area_select_image_factor)

    # work_image = np.zeros((512,512,3), np.uint8)
    base_image = np.copy(work_image)
    cv2.namedWindow(RectangleWindowTitle)
    cv2.setMouseCallback(RectangleWindowTitle, draw_rectangle)
    while 1:
        cv2.imshow(RectangleWindowTitle, work_image)
        if not cv2.EVENT_MOUSEMOVE:
            copy = work_image.copy()
            cv2.rectangle(copy, (ix, iy), (x_, y_), (0, 255, 0), 1)
            cv2.imshow(RectangleWindowTitle, copy)
        k = cv2.waitKey(1) & 0xFF
        if k == 13 and not rectangle_drawing:  # Enter: Confirm selection
            retvalue = True
            break
        elif k == 27:  # Escape: Cancel selection
            break
    cv2.destroyAllWindows()
    return retvalue


def select_cropping_area():
    global RectangleWindowTitle
    global perform_cropping
    global CropTopLeft, CropBottomRight
    global CropAreaDefined

    # Disable all buttons in main window
    button_status_change_except(0, DISABLED)
    win.update()

    RectangleWindowTitle = CropWindowTitle

    if select_rectangle_area():
        CropAreaDefined = True
        button_status_change_except(0, NORMAL)
        CropTopLeft = RectangleTopLeft
        CropBottomRight = RectangleBottomRight
        logging.debug("Crop area: (%i,%i) - (%i, %i)", CropTopLeft[0],
                      CropTopLeft[1], CropBottomRight[0], CropBottomRight[1])
    else:
        CropAreaDefined = False
        button_status_change_except(0, DISABLED)
        perform_cropping.set(False)
        perform_cropping.set(False)
        generate_video_checkbox.config(state=NORMAL if ffmpeg_installed
                                       else DISABLED)
        CropTopLeft = (0, 0)
        CropBottomRight = (0, 0)

    project_config["CropRectangle"] = CropTopLeft, CropBottomRight
    perform_cropping_checkbox.config(state=NORMAL if CropAreaDefined
                                     else DISABLED)

    # Enable all buttons in main window
    button_status_change_except(0, NORMAL)
    win.update()


def select_hole_height(work_image):
    global RectangleWindowTitle
    global perform_stabilization, perform_stabilization_checkbox
    global HoleSearchTopLeft, HoleSearchBottomRight
    global StabilizeAreaDefined
    global film_hole_height, film_hole_template, area_select_image_factor

    # Find hole height
    film_hole_height = determine_hole_height(work_image)
    if film_hole_height < 0:
        film_hole_height = 0
        StabilizeAreaDefined = False
        perform_stabilization.set(False)
        perform_stabilization_checkbox.config(state=DISABLED)
    else:
        StabilizeAreaDefined = True
        perform_stabilization_checkbox.config(state=NORMAL)
        adjust_hole_pattern_size()
    win.update()


def determine_hole_height(img):
    global film_hole_template, film_bw_template, film_wb_template

    if project_config["FilmType"] == 'R8':
        template_1 = film_wb_template
        template_2 = film_bw_template
        other_film_type = 'S8'
    else:   # S8 by default
        template_1 = film_bw_template
        template_2 = film_wb_template
        other_film_type = 'R8'
    search_img = get_image_left_stripe(img)
    top_left_1 = match_template(template_1, search_img, 230)
    top_left_2 = match_template(template_2, search_img, 230)
    if top_left_1[1] > top_left_2[1]:
        if tk.messagebox.askyesno(
                "Wrong film type detected",
                "Current project is defined to handle " + project_config["FilmType"] +
                " film type, however frames seem to be " + other_film_type + ".\r\n"
                "Do you want to change it now?"):
            film_type.set(other_film_type)
            project_config["FilmType"] = other_film_type
            set_film_type()
            top_left_aux = top_left_1
            top_left_1 = top_left_2
            top_left_2 = top_left_aux
            return 0
    logging.debug("Hole height: %i", top_left_2[1]-top_left_1[1])
    return top_left_2[1]-top_left_1[1]

def adjust_hole_pattern_size():
    global film_hole_height, film_hole_template

    if film_hole_height <= 0:
        return

    ratio = 1
    if project_config["FilmType"] == 'S8':
        ratio = film_hole_height / default_hole_height_s8
    elif project_config["FilmType"] == 'R8':
        ratio = film_hole_height / default_interhole_height_r8
    logging.debug("Hole pattern, ratio: %s, %.2f", os.path.basename(pattern_filename), ratio)
    film_hole_template = resize_image(film_hole_template, ratio*100)


def set_film_type():
    global film_type, expected_pattern_pos, pattern_filename, film_hole_template
    global default_hole_height_s8, default_interhole_height_r8
    global film_hole_height
    if project_config["FilmType"] == 'S8':
        pattern_filename = pattern_filename_s8
        expected_pattern_pos = expected_pattern_pos_s8
    elif project_config["FilmType"] == 'R8':
        pattern_filename = pattern_filename_r8
        expected_pattern_pos = expected_pattern_pos_r8

    film_type.set(project_config["FilmType"])
    film_hole_template = cv2.imread(pattern_filename, 0)
    adjust_hole_pattern_size()

    logging.debug("Film type: %s, %s, %i", project_config["FilmType"], os.path.basename(pattern_filename), film_hole_height)

    win.update()


def match_template(template, img, thres):
    w = template.shape[1]
    h = template.shape[0]
    if (w >= img.shape[1] or h >= img.shape[0]):
        logging.debug("Template (%ix%i) bigger than image  (%ix%i)",
                      w, h, img.shape[1], img.shape[0])
        return (0, 0)
    # convert img to grey
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_bw = cv2.threshold(img_grey, thres, 255, cv2.THRESH_BINARY)[1]
    res = cv2.matchTemplate(img_bw, template, cv2.TM_CCOEFF_NORMED)
    # Debug code starts
    # cv2.namedWindow('Template')
    # cv2.namedWindow('Image left strip')
    # img_bw_s = resize_image(img_bw, 50)
    # template_s = resize_image(template, 50)
    # cv2.imshow('Template', template_s)
    # cv2.imshow('Image left strip', img_bw_s)
    # cv2.waitKey(0)
    # cv2.destroyWindow('Image left strip')
    # cv2.destroyWindow('Template')
    # Debug code ends
    # Best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    return top_left


"""
###################################
Support functions for core business
###################################
"""


def display_image(img):
    global PreviewWidth, PreviewHeight
    global draw_capture_canvas, left_area_frame
    global perform_cropping

    img = resize_image(img, round(PreviewRatio*100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    DisplayableImage = ImageTk.PhotoImage(Image.fromarray(img))

    image_height = img.shape[0]
    image_width = img.shape[1]
    padding_x = 0
    padding_y = 0
    # Center only when cropping, otherwise image will shake
    if perform_cropping.get():
        if PreviewWidth > image_width:
            padding_x = round((PreviewWidth - image_width) / 2)
        if PreviewHeight > image_height:
            padding_y = round((PreviewHeight - image_height) / 2)

    draw_capture_canvas.create_image(padding_x, padding_y, anchor=NW, image=DisplayableImage)
    draw_capture_canvas.image = DisplayableImage

    win.update()


def display_image_by_frame_number(frame_number):
    global StartFrame
    # Get current file
    file = SourceDirFileList[StartFrame + frame_number]
    # read image
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)

    if project_config["PerformStabilization"]:
        img = stabilize_image(img)
    if project_config["PerformCropping"]:
        img = crop_image(img, CropTopLeft, CropBottomRight)
    else:
        img = even_image(img)

    display_image(img)

def clear_image():
    global draw_capture_canvas
    draw_capture_canvas.delete('all')


def resize_image(img, percent):
    # Calculate the proportional size of original image
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)

    dsize = (width, height)

    # resize image
    return cv2.resize(img, dsize)


def get_image_left_stripe(img):
    global HoleSearchTopLeft, HoleSearchBottomRight
    # Get partial image where the hole should be (to facilitate template search
    # by OpenCV). We do the calculations inline instead of calling the function
    # since we need the intermediate values
    # Default values defined at display initialization time, after source
    # folder is defined
    horizontal_range = (HoleSearchTopLeft[0], HoleSearchBottomRight[0])
    vertical_range = (HoleSearchTopLeft[1], HoleSearchBottomRight[1])
    return img[vertical_range[0]:vertical_range[1], horizontal_range[0]:horizontal_range[1]]


def stabilize_image(img):
    global SourceDirFileList, CurrentFrame
    global HoleSearchTopLeft, HoleSearchBottomRight
    global expected_pattern_pos, film_hole_template
    global StabilizationThreshold
    # Get image dimensions to perform image shift later
    width = img.shape[1]
    height = img.shape[0]

    left_stripe_image = get_image_left_stripe(img)

    # Search film hole pattern
    try:
        top_left = match_template(film_hole_template, left_stripe_image, float(StabilizationThreshold))
        # The coordinates returned by match template are relative to the
        # cropped image. In order to calculate the correct values to provide
        # to the translation matrix, need to convert to absolute coordinates
        top_left = (top_left[0] + HoleSearchTopLeft[0],
                    top_left[1] + HoleSearchTopLeft[1])
        # According to tests done during the development, the ideal top left
        # position for a match of the hole template used (63*339 pixels) should
        # be situated at 12% of the horizontal axis, and 38% of the vertical
        # axis. Calculate shift, according to those proportions

        move_x = round((expected_pattern_pos[0] * width / 100)) - top_left[0]
        move_y = round((expected_pattern_pos[1] * height / 100)) - top_left[1]

        logging.debug("Stabilizing frame: (%i,%i) to move (%i, %i)",
                      top_left[0], top_left[1], move_x, move_y)

        # Create the translation matrix using move_x and move_y (NumPy array)
        translation_matrix = np.array([
            [1, 0, move_x],
            [0, 1, move_y]
        ], dtype=np.float32)
        # Apply the translation to the image
        translated_image = cv2.warpAffine(src=img, M=translation_matrix,
                                          dsize=(width+move_x, height+move_y))
    except Exception as ex:
        exception_template = ("An exception of type {0} occurred. "
                              "Arguments:\n{1!r}")
        exception_details = template.format(type(ex).__name__, ex.args)
        logging.debug("Error in match_template (file %s), "
                      "returning original image. %s",
                      SourceDirFileList[CurrentFrame], exception_details)
        translated_image = img
    return translated_image


def even_image(img):
    # Get image dimensions to check whether one dimension is odd
    width = img.shape[1]
    height = img.shape[0]

    X_end = width
    Y_end = height

    # FFmpeg does not like odd dimensions
    # Adjust (decreasing BottomRight)in case of odd width/height
    if width % 2 == 1:
        X_end -= 1
    if height % 2 == 1:
        Y_end -= 1
    if X_end != 0 or Y_end != 0:
        return img[0:Y_end, 0:X_end]
    else:
        return img


def crop_image(img, top_left, botton_right):
    # Get image dimensions to perform image shift later
    width = img.shape[1]
    height = img.shape[0]

    Y_start = top_left[1]
    Y_end = min(botton_right[1], height)
    X_start = top_left[0]
    X_end = min (botton_right[0], width)

    # FFmpeg does not like odd dimensions
    # Adjust (decreasing BottomRight)in case of odd width/height
    if (X_end - X_start) % 2 == 1:
        X_end -= 1
    if (Y_end - Y_start) % 2 == 1:
        Y_end -= 1

    return img[Y_start:Y_end, X_start:X_end]


def is_ffmpeg_installed():
    global ffmpeg_installed

    cmd_ffmpeg = [FfmpegBinName, '-h']

    try:
        ffmpeg_process = sp.Popen(cmd_ffmpeg, stderr=sp.PIPE, stdout=sp.PIPE)
        ffmpeg_installed = True
    except FileNotFoundError:
        ffmpeg_installed = False
        logging.debug("ffmpeg is NOT installed.")

    return ffmpeg_installed


def get_current_dir_file_list():
    global SourceDir
    global project_config
    global SourceDirFileList
    global CurrentFrame, first_absolute_frame, last_absolute_frame
    global frame_slider
    global area_select_image_factor, screen_height
    global folder_frame_target_dir

    if not os.path.isdir(SourceDir):
        return

    SourceDirFileList = sorted(list(glob(os.path.join(
        SourceDir,
        project_config["FrameInputFilenamePattern"]))))
    if len(SourceDirFileList) == 0:
        tk.messagebox.showerror("Error!",
                                "No files match pattern name. "
                                "Please specify new one and try again")
        clear_image()
        folder_frame_target_dir.delete(0, 'end')
        return

    first_absolute_frame = int(
        ''.join(list(filter(str.isdigit,
                            os.path.basename(SourceDirFileList[0])))))
    last_absolute_frame = first_absolute_frame + len(SourceDirFileList)-1
    frame_slider.config(from_=0, to=len(SourceDirFileList)-1,
                        label='Global:'+str(CurrentFrame+first_absolute_frame))

    # In order to determine hole height, no not take the first frame, as often
    # it is not so good. Take a frame 10% ahead in the set
    sample_frame = CurrentFrame + int((len(SourceDirFileList) - CurrentFrame) * 0.1)
    work_image = cv2.imread(SourceDirFileList[sample_frame], cv2.IMREAD_UNCHANGED)
    set_hole_search_area(work_image)
    select_hole_height(work_image)
    set_film_type()
    # Select area window should be proportional to screen height
    # Deduct 120 pixels (approximately) for taskbar + window title
    area_select_image_factor = (screen_height - 200) / work_image.shape[0]
    area_select_image_factor = min(1, area_select_image_factor)

    return len(SourceDirFileList)


def valid_generated_frame_range():
    global StartFrame, frames_to_encode, first_absolute_frame

    file_count = 0
    generated_frame_list = list(glob(os.path.join(
        TargetDir, FrameCheckFilenameOutputPattern)))
    for i in range(first_absolute_frame + StartFrame,
                   first_absolute_frame + StartFrame + frames_to_encode):
        file_to_check = os.path.join(TargetDir,
                                     FrameFilenameOutputPattern % i)
        if file_to_check in generated_frame_list:
            file_count += 1
    logging.debug("Checking frame range %i-%i: %i files found",
                  first_absolute_frame + StartFrame,
                  first_absolute_frame + StartFrame + frames_to_encode,
                  file_count)

    return file_count == frames_to_encode


def set_hole_search_area(img):
    global HoleSearchTopLeft, HoleSearchBottomRight

    # Initizalize default values for perforation search area,
    # as they are relative to image size
    # Get image dimensions first
    width = img.shape[1]
    height = img.shape[0]
    # Default values are needed before the stabilization search area
    # has been defined, therefore we initialized them here
    HoleSearchTopLeft = (0, 0)
    HoleSearchBottomRight = (round(width * 0.20), height)


"""
########################
Core top level functions
########################
"""


def start_convert():
    global ConvertLoopExitRequested, ConvertLoopRunning
    global generate_video
    global video_writer
    global SourceDirFileList
    global TargetVideoFilename
    global CurrentFrame, StartFrame
    global start_from_current_frame
    global frames_to_encode
    global ffmpeg_success, ffmpeg_encoding_status

    if ConvertLoopRunning:
        ConvertLoopExitRequested = True
    else:
        if not start_from_current_frame.get():
            CurrentFrame = 0
            project_config["CurrentFrame"] = CurrentFrame
        StartFrame = CurrentFrame
        # Centralize 'frames_to_encode' update here
        if frames_to_encode_spinbox.get() == 'All':
            frames_to_encode = len(SourceDirFileList) - StartFrame
        else:
            frames_to_encode = int(frames_to_encode_spinbox.get())
            if StartFrame + frames_to_encode >= len(SourceDirFileList):
                frames_to_encode = len(SourceDirFileList) - StartFrame
        project_config["FramesToEncode"] = str(frames_to_encode)
        if frames_to_encode == 0:
            tk.messagebox.showwarning(
                "No frames match range",
                "No frames to encode.\r\n"
                "The range specified (current frame - number of frames to "
                "encode) does not match any frame.\r\n"
                "Please review your settings and try again.")
            return
        if BatchJobRunning:
            start_batch_btn.config(text="Stop batch", bg='red', fg='white')
            # Disable all buttons in main window
            button_status_change_except(start_batch_btn, DISABLED)
        else:
            Go_btn.config(text="Stop", bg='red', fg='white')
            # Disable all buttons in main window
            button_status_change_except(Go_btn, DISABLED)
        win.update()

        if project_config["GenerateVideo"]:
            TargetVideoFilename = video_filename_name.get()
            name, ext = os.path.splitext(TargetVideoFilename)
            if TargetVideoFilename == "":   # Assign default if no filename
                TargetVideoFilename = (
                    "AfterScan-" +
                    datetime.now().strftime("%Y_%m_%d-%H-%M-%S") + ".mp4")
                video_filename_name.delete(0, 'end')
                video_filename_name.insert('end', TargetVideoFilename)
            elif ext == "":
                TargetVideoFilename += ".mp4"
                video_filename_name.delete(0, 'end')
                video_filename_name.insert('end', TargetVideoFilename)
            elif os.path.isfile(os.path.join(TargetDir, TargetVideoFilename)):
                error_msg = (TargetVideoFilename + " already exist in target "
                             "folder. Overwrite?")
                if not BatchJobRunning:
                    if not tk.messagebox.askyesno("Error!", error_msg):
                        generation_exit()
                        return

        ConvertLoopRunning = True

        if not skip_frame_regeneration.get():
            win.after(1, frame_generation_loop)
        else:
            ffmpeg_success = False
            ffmpeg_encoding_status = ffmpeg_state.Pending
            win.after(1, video_generation_loop)


def generation_exit():
    global win
    global ConvertLoopExitRequested
    global ConvertLoopRunning
    global Go_btn, save_bg, save_fg
    global BatchJobRunning
    global job_list, CurrentJobEntry

    ConvertLoopExitRequested = False  # Reset flags
    ConvertLoopRunning = False
    if BatchJobRunning:
        BatchJobRunning = False  # Cancel batch jobs if any
        start_batch_btn.config(text="Start batch", bg=save_bg, fg=save_fg)
    else:
        Go_btn.config(text="Start", bg=save_bg, fg=save_fg)
    # Enable all buttons in main window
    button_status_change_except(0, NORMAL)
    win.update()

    if BatchJobRunning:
        job_list[CurrentJobEntry]['done'] = True
        job_processing_loop()


def frame_generation_loop():
    global perform_stabilization, perform_cropping
    global ConvertLoopExitRequested
    global CropTopLeft, CropBottomRight
    global TargetDir
    global CurrentFrame, first_absolute_frame
    global StartFrame, frames_to_encode
    global FrameFilenameOutputPattern
    global BatchJobRunning
    global ffmpeg_success, ffmpeg_encoding_status

    if CurrentFrame >= StartFrame + frames_to_encode:
        status_str = "Status: Frame generation OK"
        app_status_label.config(text=status_str, fg='green')
        if generate_video.get():
            ffmpeg_success = False
            ffmpeg_encoding_status = ffmpeg_state.Pending
            win.after(1, video_generation_loop)
        else:
            generation_exit()
        CurrentFrame -= 1  # Prevent being out of range
        return

    if ConvertLoopExitRequested:  # Stop button pressed
        status_str = "Status: Cancelled by user"
        app_status_label.config(text=status_str, fg='red')
        generation_exit()
        return

    # Get current file
    file = SourceDirFileList[CurrentFrame]
    # read image
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)

    if perform_stabilization.get():
        img = stabilize_image(img)
    if perform_cropping.get():
        img = crop_image(img, CropTopLeft, CropBottomRight)
    else:
        img = even_image(img)

    display_image(img)

    logging.debug("Display image: %s, target size: (%i, %i), "
                  "CurrentFrame %i", os.path.basename(file),
                  img.shape[1], img.shape[0], CurrentFrame)
    if img.shape[1] % 2 == 1 or img.shape[0] % 2 == 1:
        logging.error("Target size, one odd dimension")
        status_str = "Status: Frame %d - odd size" % CurrentFrame
        app_status_label.config(text=status_str, fg='red')
        CurrentFrame = StartFrame + frames_to_encode - 1

    if os.path.isdir(TargetDir):
        target_file = os.path.join(TargetDir, FrameFilenameOutputPattern % (first_absolute_frame + CurrentFrame))
        cv2.imwrite(target_file, img)

    frame_slider.set(CurrentFrame)
    status_str = "Status: Generating frames %.1f%%" % ((CurrentFrame-StartFrame)*100/frames_to_encode)
    app_status_label.config(text=status_str, fg='black')

    CurrentFrame += 1
    project_config["CurrentFrame"] = CurrentFrame

    win.after(1, frame_generation_loop)


def call_ffmpeg():
    global TargetDir
    global cmd_ffmpeg
    global ffmpeg_preset
    global TargetVideoFilename
    global StartFrame
    global ffmpeg_process, ffmpeg_success
    global ffmpeg_encoding_status
    global FrameFilenameOutputPattern
    global first_absolute_frame, frames_to_encode
    global CropTopLeft, CropBottomRight

    extra_input_options = []
    extra_output_options = []
    if resolution_dict[project_config["VideoResolution"]] != '':
        extra_input_options += ['-s:v',
                                str(CropBottomRight[0]-CropTopLeft[0])
                                + 'x'
                                + str(CropBottomRight[1]-CropTopLeft[1])]
    if frames_to_encode > 0:
        extra_output_options += ['-frames:v', str(frames_to_encode)]
    if ExpertMode and fill_borders.get():
        extra_output_options += [
             '-filter_complex',
             '[0:v] fillborders='
             'left=' + str(fill_borders_thickness.get()) + ':'
             'right=' + str(fill_borders_thickness.get()) + ':'
             'top=' + str(fill_borders_thickness.get()) + ':'
             'bottom=' + str(fill_borders_thickness.get()) + ':'
             'mode=' + fill_borders_mode.get() + ' [v]',
             '-map', '[v]']
    if resolution_dict[project_config["VideoResolution"]] != '':
        extra_output_options += ['-vf',
                                 'scale=' + resolution_dict[project_config["VideoResolution"]]]
    cmd_ffmpeg = [FfmpegBinName,
                  '-y',
                  '-loglevel', 'error',
                  '-stats',
                  '-flush_packets', '1',
                  '-f', 'image2',
                  '-start_number', str(StartFrame +
                                       first_absolute_frame),
                  '-framerate', str(VideoFps)]
    cmd_ffmpeg.extend(extra_input_options)
    cmd_ffmpeg.extend(
                  ['-i', os.path.join(TargetDir, FrameFilenameOutputPattern)])
    cmd_ffmpeg.extend(extra_output_options)
    cmd_ffmpeg.extend(
        ['-an',  # no audio
         '-vcodec', 'libx264',
         '-preset', ffmpeg_preset.get(),
         '-crf', '18',
         '-pix_fmt', 'yuv420p',
         os.path.join(TargetDir,
                      TargetVideoFilename)])

    logging.debug("Generated ffmpeg command: %s", cmd_ffmpeg)
    ffmpeg_process = sp.Popen(cmd_ffmpeg, stderr=sp.STDOUT,
                              stdout=sp.PIPE,
                              universal_newlines=True)
    ffmpeg_success = ffmpeg_process.wait() == 0
    ffmpeg_encoding_status = ffmpeg_state.Completed


def video_generation_loop():
    global Go_btn
    global TargetDir
    global TargetVideoFilename
    global stop_event, stop_event_lock
    global ffmpeg_success, ffmpeg_encoding_status
    global ffmpeg_process
    global frames_to_encode
    global app_status_label

    if ffmpeg_encoding_status == ffmpeg_state.Pending:
        # Check for special cases first
        if frames_to_encode == 0:
            status_str = "Status: No frames to encode"
            app_status_label.config(text=status_str, fg='red')
            tk.messagebox.showwarning(
                "No frames match range to generate video",
                "Video cannot be generated.\r\n"
                "No frames in target folder match the specified range.\r\n"
                "Please review your settings and try again.")
        elif not valid_generated_frame_range():
            status_str = "Status: No frames to encode"
            app_status_label.config(text=status_str, fg='red')
            tk.messagebox.showwarning(
                "Frames missing",
                "Video cannot be generated.\r\n"
                "Not all frames in specified range exist in target folder to "
                "allow video generation.\r\n"
                "Please regenerate frames making sure option "
                "\'Skip Frame regeneration\' is not selected, and try again.")
        else:
            logging.debug(
                "First filename in list: %s, extracted number: %s",
                os.path.basename(SourceDirFileList[0]), first_absolute_frame)
            stop_event = threading.Event()
            stop_event_lock = threading.Lock()

            ffmpeg_success = False
            ffmpeg_encoding_thread = threading.Thread(target=call_ffmpeg)
            ffmpeg_encoding_thread.daemon = True
            ffmpeg_encoding_thread.start()
            win.update()
            ffmpeg_encoding_status = ffmpeg_state.Running
            win.after(200, video_generation_loop)
    elif ffmpeg_encoding_status == ffmpeg_state.Running:
        if ConvertLoopExitRequested:
            ffmpeg_process.terminate()
            logging.info("Video generation terminated by user for %s",
                         os.path.join(TargetDir, TargetVideoFilename))
            status_str = "Status: Cancelled by user"
            app_status_label.config(text=status_str, fg='red')
            tk.messagebox.showinfo(
                "FFMPEG encoding interrupted by user",
                "\r\nVideo generation by FFMPEG has been stopped by user "
                "action.")
            generation_exit()  # Restore all settings to normal
            os.remove(os.path.join(TargetDir, TargetVideoFilename))
        else:
            line = ffmpeg_process.stdout.readline()
            if line:
                frame_str = str(line)[:-1].split()[1]
                if is_a_number(frame_str):  # Sometimes ffmpeg output might be corrupted on the way
                    encoded_frame = int(frame_str)
                    status_str = "Status: Generating video %.1f%%" % (encoded_frame*100/frames_to_encode)
                    app_status_label.config(text=status_str, fg='black')
                    display_image_by_frame_number(encoded_frame)
            win.after(200, video_generation_loop)
    elif ffmpeg_encoding_status == ffmpeg_state.Completed:
        status_str = "Status: Generating video 100%"
        app_status_label.config(text=status_str, fg='black')
        """
        stop_event.set()
        ffmpeg_progress_thread.join()
        """
        # And display results
        if ffmpeg_success:
            logging.info("Video generated OK: %s", os.path.join(TargetDir, TargetVideoFilename))
            status_str = "Status: Video generated OK"
            app_status_label.config(text=status_str, fg='green')
            if not BatchJobRunning:
                tk.messagebox.showinfo(
                    "Video generation by ffmpeg has ended",
                    "\r\nVideo encoding has finalized successfully. "
                    "You can find your video in the target folder, "
                    "as stated below\r\n" +
                    os.path.join(TargetDir, TargetVideoFilename))
        else:
            logging.info("Video generation failed for %s", os.path.join(TargetDir, TargetVideoFilename))
            status_str = "Status: Video generation failed"
            app_status_label.config(text=status_str, fg='red')
            if not BatchJobRunning:
                tk.messagebox.showinfo(
                    "FFMPEG encoding failed",
                    "\r\nVideo generation by FFMPEG has failed\r\nPlease "
                    "check the logs to determine what the problem was.")
        generation_exit()  # Restore all settings to normal


"""
###############################
Application top level functions
###############################
"""


def init_display():
    global SourceDir
    global CurrentFrame
    global SourceDirFileList
    global PreviewWidth, PreviewHeight, PreviewRatio

    # Get first file
    savedir = os.getcwd()
    if SourceDir == "":
        tk.messagebox.showerror("Error!",
                                "Please specify source and target folders.")
        return

    os.chdir(SourceDir)

    if len(SourceDirFileList) == 0:
        return

    file = SourceDirFileList[CurrentFrame]

    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)

    # Calculate preview image display ratio
    image_height = img.shape[0]
    image_width = img.shape[1]
    if abs(PreviewWidth - image_width) > abs(PreviewHeight - image_height):
        PreviewRatio = PreviewWidth/image_width
    else:
        PreviewRatio = PreviewHeight/image_height

    scale_display_update()


def afterscan_init():
    global win
    global TopWinX
    global TopWinY
    global WinInitDone
    global SourceDir
    global LogLevel
    global PreviewWidth, PreviewHeight
    global screen_height
    global ExpertMode, SmallSize

    # Initialize logging
    log_path = os.path.dirname(__file__)
    if log_path == "":
        log_path = os.getcwd()
    log_file_fullpath = log_path + "/AfterScan.debug.log"
    logging.basicConfig(
        level=LogLevel,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file_fullpath),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info("Log file: %s", log_file_fullpath)

    win = Tk()  # Create main window, store it in 'win'

    # Get screen size - maxsize gives the usable screen size
    screen_width, screen_height = win.maxsize()
    # Set dimensions of UI elements adapted to screen size
    if screen_height >= 1000:
        PreviewWidth = 700
        PreviewHeight = 560
    else:
        PreviewWidth = 620
        PreviewHeight = 460
    app_width = PreviewWidth + 370 + 30
    app_height = PreviewHeight + 170
    if ExpertMode:
        app_height += 0  # No need tto increase height for now
    if SmallSize:
        app_width -= 80
        app_height -= 60

    win.title('AfterScan ' + __version__)  # setting title of the window
    win.geometry('1080x700')  # setting the size of the window
    win.geometry('+50+50')  # setting the position of the window
    # Prevent window resize
    win.minsize(app_width, app_height)
    win.maxsize(app_width, app_height)

    win.update_idletasks()

    # Set default font size
    # Change the default Font that will affect in all the widgets
    if SmallSize:
        win.option_add("*font", "TkDefaultFont 8")
    else:
        win.option_add("*font", "TkDefaultFont 10")
    win.resizable(False, False)

    # Get Top window coordinates
    TopWinX = win.winfo_x()
    TopWinY = win.winfo_y()

    WinInitDone = True

    logging.debug("AfterScan initialized")


def build_ui():
    global win
    global SourceDir, TargetDir
    global folder_frame_source_dir, folder_frame_target_dir
    global perform_cropping, cropping_btn
    global generate_video, generate_video_checkbox
    global fill_borders, fill_borders_checkbox
    global fill_borders_thickness, fill_borders_thickness_slider
    global fill_borders_thickness_slider, fill_borders_mode_label
    global fill_borders_mode_label_dropdown, fill_borders_mode
    global start_from_current_frame
    global frames_to_encode_spinbox, frames_to_encode_str, frames_to_encode
    global save_bg, save_fg
    global source_folder_btn, target_folder_btn
    global perform_stabilization, perform_stabilization_checkbox
    global stabilization_threshold_spinbox, stabilization_threshold_str
    global StabilizationThreshold
    global perform_cropping_checkbox, Crop_btn
    global Go_btn
    global Exit_btn
    global video_fps_dropdown_selected
    global video_fps_dropdown, video_fps_label, video_filename_name
    global resolution_dropdown, resolution_label, resolution_dropdown_selected
    global ffmpeg_preset
    global ffmpeg_preset_rb1, ffmpeg_preset_rb2, ffmpeg_preset_rb3
    global skip_frame_regeneration
    global ExpertMode
    global pattern_filename
    global frame_input_filename_pattern
    global frame_slider, CurrentFrame
    global film_type, film_hole_template
    global job_list_listbox
    global app_status_label
    global PreviewWidth, PreviewHeight
    global left_area_frame
    global draw_capture_canvas
    global custom_ffmpeg_path
    global project_config
    global start_batch_btn, add_job_btn, delete_job_btn

    # Create a frame to add a border to the preview
    left_area_frame = Frame(win)
    #left_area_frame.grid(row=0, column=0, padx=5, pady=5, sticky=N)
    left_area_frame.pack(side=LEFT, padx=5, pady=5, anchor=N)
    draw_capture_canvas = Canvas(left_area_frame, bg='dark grey',
                                 width=PreviewWidth, height=PreviewHeight)
    draw_capture_canvas.pack(side=TOP, anchor=N)

    # Frame for standard widgets
    right_area_frame = Frame(win, width=320, height=450)
    #right_area_frame.grid(row=0, column=1, rowspan=2, padx=5, pady=5, sticky=N)
    right_area_frame.pack(side=LEFT, padx=5, pady=5, anchor=N)

    # Frame for top section of standard widgets ******************************
    regular_top_section_frame = Frame(right_area_frame, width=50, height=50)
    regular_top_section_frame.pack(side=TOP, padx=2, pady=2)

    # Create frame to display current frame and slider
    frame_frame = LabelFrame(regular_top_section_frame, text='Current frame',
                               width=35, height=10)
    frame_frame.grid(row=0, column=0, sticky=W)

    frame_selected = IntVar()
    frame_slider = Scale(frame_frame, orient=HORIZONTAL, from_=0, to=0,
                         variable=frame_selected, command=select_scale_frame,
                         length=120, label='Global:',
                         highlightthickness=1, takefocus=1)
    frame_slider.pack(side=BOTTOM, ipady=4)
    frame_slider.set(CurrentFrame)

    # Application status label
    app_status_label = Label(regular_top_section_frame, width=46, borderwidth=2,
                             relief="groove", text='Status: Idle',
                             highlightthickness=1)
    app_status_label.grid(row=1, column=0, columnspan=3, sticky=W,
                          pady=5)

    # Application Exit button
    Exit_btn = Button(regular_top_section_frame, text="Exit", width=10,
                      height=5, command=exit_app, activebackground='red',
                      activeforeground='white', wraplength=80)
    Exit_btn.grid(row=0, column=1, sticky=W, padx=5)


    # Application start button
    Go_btn = Button(regular_top_section_frame, text="Start", width=12, height=5,
                    command=start_convert, activebackground='green',
                    activeforeground='white', wraplength=80)
    Go_btn.grid(row=0, column=2, sticky=W)

    # Create frame to select source and target folders *******************************
    folder_frame = LabelFrame(right_area_frame, text='Folder selection', width=50,
                              height=8)
    folder_frame.pack(side=TOP, padx=2, pady=2, anchor=W)

    source_folder_frame = Frame(folder_frame)
    source_folder_frame.pack(side=TOP)
    folder_frame_source_dir = Entry(source_folder_frame, width=36,
                                    borderwidth=1)
    folder_frame_source_dir.pack(side=LEFT)
    folder_frame_source_dir.delete(0, 'end')
    folder_frame_source_dir.insert('end', SourceDir)

    source_folder_btn = Button(source_folder_frame, text='Source', width=6,
                               height=1, command=set_source_folder,
                               activebackground='green',
                               activeforeground='white', wraplength=80)
    source_folder_btn.pack(side=LEFT)

    target_folder_frame = Frame(folder_frame)
    target_folder_frame.pack(side=TOP)
    folder_frame_target_dir = Entry(target_folder_frame, width=36,
                                    borderwidth=1)
    folder_frame_target_dir.pack(side=LEFT)
    target_folder_btn = Button(target_folder_frame, text='Target', width=6,
                               height=1, command=set_target_folder,
                               activebackground='green',
                               activeforeground='white', wraplength=80)
    target_folder_btn.pack(side=LEFT)

    save_bg = source_folder_btn['bg']
    save_fg = source_folder_btn['fg']

    folder_bottom_frame = Frame(folder_frame)
    folder_bottom_frame.pack(side=BOTTOM, ipady=2)

    frame_filename_pattern_frame = Frame(folder_frame)
    frame_filename_pattern_frame.pack(side=TOP, anchor=W)
    frame_filename_pattern_label = Label(frame_filename_pattern_frame,
                                         text='Frame input filename pattern:')
    frame_filename_pattern_label.pack(side=LEFT, anchor=W)
    frame_input_filename_pattern = Entry(frame_filename_pattern_frame,
                                         width=20, borderwidth=1)
    frame_input_filename_pattern.bind("<KeyRelease>", frame_input_filename_pattern_focus_out)
    frame_input_filename_pattern.pack(side=LEFT, anchor=W)
    frame_input_filename_pattern.delete(0, 'end')
    frame_input_filename_pattern.insert('end', project_config["FrameInputFilenamePattern"])

    # Define post-processing area *********************************************
    postprocessing_frame = LabelFrame(right_area_frame,
                                      text='Frame post-processing',
                                      width=40, height=8)
    postprocessing_frame.pack(side=TOP, padx=2, pady=2)
    postprocessing_row = 0

    # Check box to select start from current frame
    start_from_current_frame = tk.BooleanVar(value=False)
    start_from_current_frame_checkbox = tk.Checkbutton(
        postprocessing_frame, text='Start from current frame',
        variable=start_from_current_frame, onvalue=True, offvalue=False,
        command=start_from_current_frame_selection, width=20)
    start_from_current_frame_checkbox.grid(row=postprocessing_row, column=0,
                                           columnspan=3, sticky=W)
    postprocessing_row += 1

    # Spinbox to select number of frames to process
    frames_to_encode_label = tk.Label(postprocessing_frame,
                                      text='Frames to encode:',
                                      width=16)
    frames_to_encode_label.grid(row=postprocessing_row, column=0,
                                columnspan=2, sticky=W)
    frames_to_encode_str = tk.StringVar(value=str(frames_to_encode))
    frames_to_encode_selection_aux = postprocessing_frame.register(
        frames_to_encode_selection)
    frames_to_encode_spinbox = tk.Spinbox(
        postprocessing_frame,
        command=(frames_to_encode_selection_aux, '%d'), width=8,
        textvariable=frames_to_encode_str, from_=0, to=50000)
    frames_to_encode_spinbox.grid(row=postprocessing_row, column=2, sticky=W)
    frames_to_encode_spinbox.bind("<KeyRelease>", frames_to_encode_spinbox_focus_out)
    frames_to_encode_selection('down')
    postprocessing_row += 1

    # Check box to do stabilization or not
    perform_stabilization = tk.BooleanVar(value=False)
    perform_stabilization_checkbox = tk.Checkbutton(
        postprocessing_frame, text='Stabilize threshold',
        variable=perform_stabilization, onvalue=True, offvalue=False, width=15,
        command=perform_stabilization_selection)
    perform_stabilization_checkbox.grid(row=postprocessing_row, column=0,
                                        columnspan=2, sticky=W)
    perform_stabilization_checkbox.config(state=DISABLED)

    # Spinbox to select stabilization threshold
    stabilization_threshold_str = tk.StringVar(value=str(StabilizationThreshold))
    stabilization_threshold_selection_aux = postprocessing_frame.register(
        stabilization_threshold_selection)
    stabilization_threshold_spinbox = tk.Spinbox(
        postprocessing_frame,
        command=(stabilization_threshold_selection_aux, '%d'), width=8,
        textvariable=stabilization_threshold_str, from_=0, to=255)
    stabilization_threshold_spinbox.grid(row=postprocessing_row, column=2, sticky=W)
    stabilization_threshold_spinbox.bind("<KeyRelease>", stabilization_threshold_spinbox_focus_out)
    stabilization_threshold_selection('down')
    postprocessing_row += 1

    # Check box to do cropping or not
    perform_cropping = tk.BooleanVar(value=False)
    perform_cropping_checkbox = tk.Checkbutton(
        postprocessing_frame, text='Crop', variable=perform_cropping,
        onvalue=True, offvalue=False, command=perform_cropping_selection,
        width=4)
    perform_cropping_checkbox.grid(row=postprocessing_row, column=0, sticky=W)
    perform_cropping_checkbox.config(state=DISABLED)
    cropping_btn = Button(postprocessing_frame, text='Define crop area',
                          width=18, height=1, command=select_cropping_area,
                          activebackground='green', activeforeground='white',
                          wraplength=120)
    cropping_btn.grid(row=postprocessing_row, column=1, columnspan=2, sticky=W)
    postprocessing_row += 1

    # Radio buttons to select R8/S8. Required to select adequate pattern, and match position
    film_type = StringVar()
    film_type_S8_rb = Radiobutton(postprocessing_frame, text="Super 8", command=set_film_type,
                                  variable=film_type, value='S8')
    film_type_S8_rb.grid(row=postprocessing_row, column=0, sticky=W)
    film_type_R8_rb = Radiobutton(postprocessing_frame, text="Regular 8", command=set_film_type,
                                  variable=film_type, value='R8')
    film_type_R8_rb.grid(row=postprocessing_row, column=1, sticky=W)
    film_type.set(project_config["FilmType"])

    # Define video generating area ************************************
    video_frame = LabelFrame(right_area_frame,
                             text='Video generation',
                             width=50, height=8)
    video_frame.pack(side=TOP, padx=2, pady=2)
    video_row = 0

    # Check box to generate video or not
    generate_video = tk.BooleanVar(value=False)
    generate_video_checkbox = tk.Checkbutton(video_frame,
                                             text='Video',
                                             variable=generate_video,
                                             onvalue=True, offvalue=False,
                                             command=generate_video_selection,
                                             width=5)
    generate_video_checkbox.grid(row=video_row, column=0, sticky=W)
    generate_video_checkbox.config(state=NORMAL if ffmpeg_installed
                                   else DISABLED)
    # Check box to skip frame regeneration
    skip_frame_regeneration = tk.BooleanVar(value=False)
    skip_frame_regeneration_cb = tk.Checkbutton(
        video_frame, text='Skip Frame regeneration',
        variable=skip_frame_regeneration, onvalue=True, offvalue=False,
        width=20)
    skip_frame_regeneration_cb.grid(row=video_row, column=1,
                                    columnspan=2, sticky=W)
    skip_frame_regeneration_cb.config(state=NORMAL if ffmpeg_installed
                                      else DISABLED)
    video_row += 1

    # Video filename
    video_filename_label = Label(video_frame, text='Video filename:')
    video_filename_label.grid(row=video_row, column=0, sticky=W)
    video_filename_name = Entry(video_frame, width=26, borderwidth=1)
    video_filename_name.grid(row=video_row, column=1, columnspan=2,
                             sticky=W)
    video_filename_name.delete(0, 'end')
    video_filename_name.insert('end', TargetVideoFilename)
    video_row += 1

    # Drop down to select FPS
    # Dropdown menu options
    fps_list = [
        "16",
        "16.67",
        "18",
        "24",
        "25",
        "29.97",
        "30",
        "48",
        "50"
    ]

    # datatype of menu text
    video_fps_dropdown_selected = StringVar()

    # initial menu text
    video_fps_dropdown_selected.set("18")

    # Create FPS Dropdown menu
    video_fps_frame = Frame(video_frame)
    video_fps_frame.grid(row=video_row, column=0, sticky=W)
    video_fps_label = Label(video_fps_frame, text='FPS:')
    video_fps_label.pack(side=LEFT, anchor=W)
    video_fps_label.config(state=DISABLED)
    video_fps_dropdown = OptionMenu(video_fps_frame,
                                    video_fps_dropdown_selected, *fps_list,
                                    command=set_fps)
    video_fps_dropdown.config(takefocus=1)
    video_fps_dropdown.pack(side=LEFT, anchor=E)
    video_fps_dropdown.config(state=DISABLED)

    # Create FFmpeg preset options
    ffmpeg_preset_frame = Frame(video_frame)
    ffmpeg_preset_frame.grid(row=video_row, column=1, columnspan=2,
                             sticky=W)
    ffmpeg_preset = StringVar()
    ffmpeg_preset_rb1 = Radiobutton(ffmpeg_preset_frame,
                                    text="Best quality (slow)",
                                    variable=ffmpeg_preset, value='veryslow')
    ffmpeg_preset_rb1.pack(side=TOP, anchor=W)
    ffmpeg_preset_rb1.config(state=DISABLED)
    ffmpeg_preset_rb2 = Radiobutton(ffmpeg_preset_frame, text="Medium",
                                    variable=ffmpeg_preset, value='medium')
    ffmpeg_preset_rb2.pack(side=TOP, anchor=W)
    ffmpeg_preset_rb2.config(state=DISABLED)
    ffmpeg_preset_rb3 = Radiobutton(ffmpeg_preset_frame,
                                    text="Fast (low quality)",
                                    variable=ffmpeg_preset, value='veryfast')
    ffmpeg_preset_rb3.pack(side=TOP, anchor=W)
    ffmpeg_preset_rb3.config(state=DISABLED)
    ffmpeg_preset.set('medium')
    video_row += 1

    # Drop down to select resolution
    # datatype of menu text
    resolution_dropdown_selected = StringVar()

    # initial menu text
    resolution_dropdown_selected.set("Unchanged")

    # Create resolution Dropdown menu
    resolution_frame = Frame(video_frame)
    resolution_frame.grid(row=video_row, column=0, columnspan= 2, sticky=W)
    resolution_label = Label(resolution_frame, text='Resolution:')
    resolution_label.pack(side=LEFT, anchor=W)
    resolution_label.config(state=DISABLED)
    resolution_dropdown = OptionMenu(resolution_frame,
                                    resolution_dropdown_selected, *resolution_dict.keys(),
                                    command=set_resolution)
    resolution_dropdown.config(takefocus=1)
    resolution_dropdown.pack(side=LEFT, anchor=E)
    resolution_dropdown.config(state=DISABLED)
    video_row += 1

    # Define job list area ***************************************************
    job_list_frame = LabelFrame(left_area_frame,
                             text='Job List',
                             width=50, height=8)
    job_list_frame.pack(side=TOP, padx=2, pady=2, anchor=W)
    job_list_row = 0

    # job listbox
    job_list_listbox = Listbox(job_list_frame, width=50, height=5)
    job_list_listbox.grid(column=0, row=0, padx=5, pady=2)

    # job listbox scrollbars
    job_list_listbox_scrollbar_y = Scrollbar(job_list_frame, orient="vertical")
    job_list_listbox_scrollbar_y.config(command=job_list_listbox.yview)
    job_list_listbox_scrollbar_y.grid(row=0, column=1, sticky=NS)
    job_list_listbox_scrollbar_x = Scrollbar(job_list_frame, orient="horizontal")
    job_list_listbox_scrollbar_x.config(command=job_list_listbox.xview)
    job_list_listbox_scrollbar_x.grid(row=1, column=0, columnspan=1, sticky=EW)

    job_list_listbox.config(xscrollcommand=job_list_listbox_scrollbar_x.set)
    job_list_listbox.config(yscrollcommand=job_list_listbox_scrollbar_y.set)

    # Define job list button area
    job_list_btn_frame = Frame(job_list_frame,
                             width=50, height=8)
    job_list_btn_frame.grid(row=0, column=2, padx=2, pady=2, sticky=W)

    # Add job button
    add_job_btn = Button(job_list_btn_frame, text="Add job", width=12, height=1,
                    command=job_list_add_current, activebackground='green',
                    activeforeground='white', wraplength=100)
    add_job_btn.pack(side=TOP, padx=2, pady=2)

    # Delete job button
    delete_job_btn = Button(job_list_btn_frame, text="Delete job", width=12, height=1,
                    command=job_list_delete_selected, activebackground='green',
                    activeforeground='white', wraplength=100)
    delete_job_btn.pack(side=TOP, padx=2, pady=2)

    # Start processing job button
    start_batch_btn = Button(job_list_btn_frame, text="Start batch", width=12, height=1,
                    command=start_processing_job_list, activebackground='green',
                    activeforeground='white', wraplength=100)
    start_batch_btn.pack(side=TOP, padx=2, pady=2)

    postprocessing_bottom_frame = Frame(video_frame, width=30)
    postprocessing_bottom_frame.grid(row=video_row, column=0)

    if ExpertMode:
        # Frame for expert widgets
        #expert_frame = Frame(win, width=900, height=150)
        #expert_frame.grid(row=1, column=0, padx=5, pady=5, sticky=NW)

        # Custom ffmpeg path
        custom_ffmpeg_path_frame = LabelFrame(right_area_frame, text='Custom FFMpeg path',
                                     width=26, height=8)
        custom_ffmpeg_path_frame.pack(side=TOP)
        custom_ffmpeg_path = Entry(custom_ffmpeg_path_frame, width=26, borderwidth=1)
        custom_ffmpeg_path.pack()
        custom_ffmpeg_path.delete(0, 'end')
        custom_ffmpeg_path.insert('end', FfmpegBinName)
        custom_ffmpeg_path.bind("<KeyRelease>", custom_ffmpeg_path_focus_out)

        # Video filters area
        video_filters_frame = LabelFrame(right_area_frame, text='Video Filters Area',
                                     width=26, height=8)
        video_filters_frame.pack(side=TOP)

        # Check box - Fill borders
        fill_borders = tk.BooleanVar(value=False)
        fill_borders_checkbox = tk.Checkbutton(video_filters_frame,
                                               text='Fill borders',
                                               variable=fill_borders,
                                               onvalue=True, offvalue=False,
                                               command=fill_borders_selection,
                                               width=9)
        fill_borders_checkbox.grid(row=0, column=0, columnspan=1, sticky=W)
        fill_borders_checkbox.config(state=NORMAL if ffmpeg_installed and
                                     perform_cropping.get() else DISABLED)
        # Fill border thickness
        fill_borders_thickness = IntVar(value=20)
        fill_borders_thickness_slider = Scale(
            video_filters_frame, orient=HORIZONTAL, from_=5, to=50,
            variable=fill_borders_thickness,
            command=fill_borders_set_thickness_scale, length=80)
        fill_borders_thickness_slider.grid(row=0, column=1, sticky=W)
        fill_borders_thickness_slider.config(state=DISABLED)
        # Fill border mode
        # Dropdown menu options
        fill_borders_mode_list = [
            "smear",
            "mirror",
            "fixed"
        ]

        # datatype of menu text
        fill_borders_mode = StringVar()

        # initial menu text
        fill_borders_mode.set("smear")

        # Create fill border mode Dropdown menu
        fill_borders_mode_frame = Frame(video_filters_frame)
        fill_borders_mode_frame.grid(row=0, column=2, sticky=W)
        fill_borders_mode_label = Label(fill_borders_mode_frame, text='Mode:')
        fill_borders_mode_label.pack(side=LEFT, anchor=W)
        fill_borders_mode_label.config(state=DISABLED)
        fill_borders_mode_label_dropdown = OptionMenu(
            fill_borders_mode_frame,
            fill_borders_mode,
            *fill_borders_mode_list,
            command=fill_borders_set_mode)
        fill_borders_mode_label_dropdown.pack(side=LEFT, anchor=E)
        fill_borders_mode_label_dropdown.config(state=DISABLED)

        video_row += 1


def exit_app():  # Exit Application
    global win
    save_general_config()
    save_project_config()
    save_job_list()
    win.destroy()


def main(argv):
    global LogLevel, LoggingMode
    global film_hole_template, film_bw_template, film_wb_template
    global ExpertMode
    global FfmpegBinName
    global IsWindows, IsLinux, IsMac
    global pattern_filename
    global project_config_filename, project_config_basename
    global perform_stabilization
    global ui_init_done
    global IgnoreConfig
    global job_list
    global project_settings
    global SmallSize

    LoggingMode = "warning"

    # Create job dictionary
    job_list = {}

    # Create project setttings dictionary
    project_settings = {}

    pattern_filenames = [pattern_filename, pattern_bw_filename, pattern_wb_filename]
    for filename in pattern_filenames:
        if not os.path.isfile(filename):
            tk.messagebox.showerror(
                "Error: Hole template not found",
                "After scan needs film hole templates to work.\r\n"
                "File " + os.path.basename(filename) +
                " does not exist; Please copy it to the working folder of "
                "AfterScan and try again.")
            exit(-1)
    film_hole_template = cv2.imread(pattern_filename, 0)
    film_bw_template =  cv2.imread(pattern_bw_filename, 0)
    film_wb_template =  cv2.imread(pattern_wb_filename, 0)

    opts, args = getopt.getopt(argv, "shiel:")

    for opt, arg in opts:
        if opt == '-l':
            LoggingMode = arg
        elif opt == '-e':
            ExpertMode = True
        elif opt == '-s':
            SmallSize = True
        elif opt == '-i':
            IgnoreConfig = True
        elif opt == '-h':
            print("AfterScan")
            print("  -l <log mode>  Set log level:")
            print("      <log mode> = [DEBUG|INFO|WARNING|ERROR]")
            print("  -e             Enable expert mode")
            print("  -i             Ignore existing config")
            print("  -s             Smaller font")
            exit()

    LogLevel = getattr(logging, LoggingMode.upper(), None)
    if not isinstance(LogLevel, int):
        raise ValueError('Invalid log level: %s' % LogLevel)

    afterscan_init()

    load_general_config()

    ffmpeg_installed = False
    if platform.system() == 'Windows':
        IsWindows = True
        FfmpegBinName = 'C:\\ffmpeg\\bin\\ffmpeg.exe'
        AltFfmpegBinName = 'ffmpeg.exe'
        logging.debug("Detected Windows OS")
    elif platform.system() == 'Linux':
        IsLinux = True
        FfmpegBinName = 'ffmpeg'
        AltFfmpegBinName = 'ffmpeg'
        logging.debug("Detected Linux OS")
    elif platform.system() == 'Darwin':
        IsMac = True
        FfmpegBinName = 'ffmpeg'
        AltFfmpegBinName = 'ffmpeg'
        logging.debug("Detected Darwin (MacOS) OS")
    else:
        FfmpegBinName = 'ffmpeg'
        AltFfmpegBinName = 'ffmpeg'
        logging.debug("OS not recognized: " + platform.system())

    if is_ffmpeg_installed():
        ffmpeg_installed = True
    elif platform.system() == 'Windows':
        FfmpegBinName = AltFfmpegBinName
        if is_ffmpeg_installed():
            ffmpeg_installed = True
    if not ffmpeg_installed:
        tk.messagebox.showerror(
            "Error: ffmpeg is not installed",
            "FFmpeg is not installed in this computer.\r\n"
            "It is not mandatory for the application to run; "
            "Frame stabilization and cropping will still work, "
            "video generation will not")

    build_ui()

    if SourceDir is not None:
        project_config_filename = os.path.join(SourceDir,
                                               project_config_basename)
    load_project_settings()
    
    load_project_config()
    decode_project_config()

    load_job_list()

    get_current_dir_file_list()

    ui_init_done = True

    # Disable a few items that should be not operational without source folder
    if len(SourceDir) == 0:
        Go_btn.config(state=DISABLED)
        cropping_btn.config(state=DISABLED)
        frame_slider.config(state=DISABLED)
    else:
        frame_slider.set(CurrentFrame)

    init_display()

    # Main Loop
    win.mainloop()  # running the loop that works as a trigger


if __name__ == '__main__':
    main(sys.argv[1:])
