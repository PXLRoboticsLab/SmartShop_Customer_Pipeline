#!/usr/bin/env python
import Tkinter as tk
import cv2
import os
import tensorflow as tf
import align.detect_face
import facenet
import tkMessageBox
import argparse
import time
import shutil
import threading
import change_name_dialog
import rospy
import numpy as np
from scipy import misc
from Tkinter import *
from PIL import Image, ImageTk
from std_msgs.msg import String

scan_list = []
scanning = False
nrof_images = 1


def mark_faces(frame):
    height, width, _ = frame.shape
    faces, boxes = align_data([frame], 160, 44, pnet, rnet, onet)
    if boxes is not None:
        for box in boxes:
            if box[4] > 0.95:
                frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                frame = cv2.putText(frame, "{:.2%}".format(box[4]), (int(box[0]), int(box[1] - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
                if scanning:
                    frame = cv2.rectangle(frame, (int((width / 2) - 201), 9), (int((width / 2) + 201), 31),
                                          (0, 51, 153), 2)
                    frame = cv2.rectangle(frame, (int((width / 2) - 200), 10),
                                          (int((width / 2) + ((20 * nrof_images) - 200)), 30), (102, 153, 255),
                                          cv2.FILLED)
    return frame, faces


def align_data(image_list, image_size, margin, pnet, rnet, onet):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    img_list = []
    boxes = []

    for x in xrange(len(image_list)):
        img_size = np.asarray(image_list[x].shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(image_list[x], minsize, pnet, rnet, onet, threshold, factor)
        nrof_samples = len(bounding_boxes)
        if nrof_samples > 0:
            for i in xrange(nrof_samples):
                if bounding_boxes[i][4] > 0.95:
                    det = np.squeeze(bounding_boxes[i, 0:4])
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0] - margin / 2, 0)
                    bb[1] = np.maximum(det[1] - margin / 2, 0)
                    bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                    bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                    cropped = image_list[x][bb[1]:bb[3], bb[0]:bb[2], :]
                    # Check if found face isn't to blurry
                    if cv2.Laplacian(cropped, cv2.CV_64F).var() > 100:
                        aligned = misc.imresize(cropped, (image_size, image_size), interp="bilinear")
                        prewhitened = facenet.prewhiten(aligned)
                        img_list.append(prewhitened)
                        boxes.append(bounding_boxes[i])

    if len(img_list) > 0:
        images = np.stack(img_list)
        return images, boxes

    return None, None


def load_labels_from_folder(folder):
    labels = []
    for dir in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, dir)):
            if dir != "Unknown":
                labels.append(dir)
    return labels


def create_network_face_detection():
    gpu_memory_fraction = 0.20
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    return pnet, rnet, onet


def fill_listbox():
    selection = ""
    if list_box.curselection():
        selection = list_box.get(list_box.curselection()[0])
    list_box.delete(0, END)
    for label in load_labels_from_folder(folder):
        list_box.insert(END, label.replace("_", " "))

    if selection != "":
        for i in range(list_box.size()):
            if list_box.get(i) == selection:
                list_box.select_set(i)
                break

    root.after(10000, fill_listbox)


def show_frame():
    global scan_list
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2image = cv2.resize(cv2image, (750, 500))
    cv2image, scan_list = mark_faces(cv2image)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)


def add_customer():
    name = name_entry.get()
    if name is not "":
        scan_face(name)
    else:
        tkMessageBox.showerror("Error!", "Please enter a name.")


def remove_customer():
    if list_box.curselection():
        result = tkMessageBox.askquestion("Delete person?",
                                          "Are you sure you want to delete " + list_box.get(
                                              list_box.curselection()[0]) + "?",
                                          icon='warning')
        if result == "yes":
            shutil.rmtree(os.path.join(folder, list_box.get(list_box.curselection()[0]).replace(" ", "_")))
            list_box.delete(list_box.curselection()[0])
            time.sleep(1)
            pub.publish("Train")
    else:
        tkMessageBox.showerror("Error!", "No client selected.")


def scan_face_thread(name):
    global train
    global nrof_images
    global scanning
    scanning = True

    nrof_images = 1
    path = os.path.join(folder, name)

    while nrof_images < 21:
        if scan_list is not None:
            misc.imsave(os.path.join(path, name + '_' + str('%0*d' % (4, nrof_images)) + '.png'), scan_list[0])
            nrof_images += 1
        time.sleep(0.5)
    scanning = False
    tkMessageBox.showinfo("Scan done", "Scanning of customer face is done!")
    pub.publish("Train")


def scan_face(name):
    global scanning
    scanning = False
    while not scanning:
        if scan_list is not None:
            list_box.insert(END, name)
            list_box.select_set(END)
            name_entry.delete(0, "end")
            tkMessageBox.showinfo("Starting scan.", "Starting face scan. This will take approximately 10 seconds."
                                                    "\nDuring the scan rotate your head slightly to the left and right."
                                                    "\nPress OK to continue.")
            os.mkdir(os.path.join(folder, name.replace(" ", "_")))
            threading.Thread(target=scan_face_thread, args=(name.replace(' ', '_'), )).start()
        else:
            tkMessageBox.showerror("Error!", "There is no face in the current frame."
                                             "\nPress OK and try again.")


def show_dialog():
    if list_box.curselection():
        inputDialog = change_name_dialog.MyDialog(root, folder, list_box.get(list_box.curselection()[0]))
        root.wait_window(inputDialog.top)
        fill_listbox()
    else:
        tkMessageBox.showerror("Error!", "No client selected.")


if __name__ == "__main__":
    rospy.init_node("customer_dashboard")
    pub = rospy.Publisher("train_command", String, queue_size=1)

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,
                        help='Path to the data directory containing classifier data.')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.20)

    args = parser.parse_args()

    folder = args.data_dir
    print("Loading face detection model")
    pnet, rnet, onet = create_network_face_detection()

    width, height = 1920, 1080
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    root = tk.Tk()
    root.title("Customer dashboard")
    #root.wm_iconbitmap("@/home/maarten/Pictures/icon2.xbm")
    root.resizable(width=False, height=False)
    root.geometry("{}x{}".format(1000, 500))
    root.bind("<Escape>", lambda e: root.quit())

    left_container = Frame(root, bg="darkgrey", width=300, height=500)
    center_container = Frame(root, bg="darkgrey", width=700, height=500)

    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)

    left_container.grid(row=0, column=0, sticky="nsew")
    center_container.grid(row=0, column=1, sticky="nsew")

    left_container.grid_rowconfigure(5, weight=height)
    left_container.grid_columnconfigure(1, weight=width)

    list_box = Listbox(left_container, height=20)
    fill_listbox()
    list_box.grid(row=2, columnspan=2, padx=10, pady=10, sticky="nsew")

    remove_button = Button(left_container, text="Remove selected client", command=remove_customer)
    name_label = Label(left_container, text="Name:")
    name_entry = Entry(left_container)
    add_button = Button(left_container, text="Add new client", command=add_customer)
    change_button = Button(left_container, text="Change selected client name.", command=show_dialog)

    remove_button.grid(row=3, columnspan=2, padx=10, sticky="nsew")
    name_label.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="nsew")
    name_entry.grid(row=0, column=1, padx=(5, 10), pady=10, sticky="ew")
    add_button.grid(row=1, columnspan=2, padx=10, sticky="nsew")
    change_button.grid(row=4, columnspan=2, padx=10, pady=10, sticky="nsew")

    lmain = Label(center_container, bg="darkgrey")
    lmain.pack(fill=BOTH, expand=True, padx=10, pady=10)

    show_frame()
    root.mainloop()
