import cv2
import os
import numpy as np
from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox

label_names = {}

def create_capture_frame():
    main_frame.destroy()
    global capture_faces_frame
    capture_faces_frame = Frame(root)
    capture_faces_frame.pack(pady=10)
    name_label = Label(capture_faces_frame, text="Enter your name:")
    name_label.pack(pady=10)
    global name_entry
    name_entry = Entry(capture_faces_frame)
    name_entry.pack(pady=10)
    name_entry.focus()
    capture_button = Button(capture_faces_frame, text="Capture", command=capture_faces)
    capture_button.pack(pady=10)
    back_button = Button(capture_faces_frame, text="Back", command=exit)
    back_button.pack(pady=10)

def create_main_frame():
    if 'capture_faces_frame' in globals():
        capture_faces_frame.destroy()
    global main_frame
    main_frame = Frame(root)
    main_frame.pack(pady=10)
    capture_button = Button(main_frame, text="Capture Face", width=15, command=create_capture_frame)
    capture_button.pack(pady=10)
    train_button = Button(main_frame, text="Train Model", width=15, command=train_model)
    train_button.pack(pady=10)
    recognize_button = Button(main_frame, text="Recognize Face", width=15, command=recognize_faces)
    recognize_button.pack(pady=10)
    exit_button = Button(main_frame, text="Exit", width=15, command=root.destroy)
    exit_button.pack(pady=10)

def capture_faces():
    name = name_entry.get()
    if name == "":
        messagebox.showerror("Error", "Please enter your name")
        name_entry.focus()
        return
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, frame = cap.read()
        if ret == False:
            continue
        count += 1
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_frame, 1.1, 4)
        captured_face = None
        for (x, y, w, h) in faces:
            captured_face = frame[y: y + w, x: x + h]
            break
        if captured_face is None:
            print("No face detected in this frame")
            continue
        cropped_face = cv2.resize(captured_face, (250, 250))
        gray_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
        faces_collection_directory = 'faces'
        if not os.path.exists(faces_collection_directory):
            os.mkdir(faces_collection_directory)
        face_path = os.path.join(faces_collection_directory, name + "_" + str(count) + ".jpg")
        cv2.imwrite(face_path, gray_face)
        cv2.putText(gray_face, str(count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (1, 1, 1), 3)
        cv2.imshow("Face", gray_face)
        if cv2.waitKey(1) == 27 or count == 100:
            break
    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Success", "Collected " + str(count) + " images of " + name + " successfully!")
    capture_faces_frame.destroy()
    create_main_frame()

def train_model():
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_recognizer_model = cv2.face.LBPHFaceRecognizer_create()
    faces_directory = 'faces'
    faces = []
    labels = []
    name_to_label = {}
    current_label = 1
    for face_file_name in os.listdir(faces_directory):
        name = face_file_name.split('_')[0]
        if name not in name_to_label:
            name_to_label[name] = current_label
            current_label += 1
        face_path = os.path.join(faces_directory, face_file_name)
        face = cv2.imread(face_path, cv2.IMREAD_GRAYSCALE)
        faces.append(face)
        labels.append(name_to_label[name])
    face_recognizer_model.train(faces, np.array(labels))
    face_recognizer_model.save('trained_data.xml')
    global label_names
    label_names = {v: k for k, v in name_to_label.items()}
    cv2.destroyAllWindows()
    messagebox.showinfo("Success", "Model trained successfully!")

def recognize_faces():
    global label_names
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_classifier.empty():
        messagebox.showerror("Error", "Could not load face detection model")
        return
    face_recognizer_model = cv2.face.LBPHFaceRecognizer_create()
    if not os.path.exists('trained_data.xml'):
        messagebox.showerror("Error", "No trained model found. Please train the model first.")
        return
    try:
        face_recognizer_model.read('trained_data.xml')
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model: {str(e)}")
        return
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            for i in [1, 2, 3]:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    break
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open any camera")
                return
    except Exception as e:
        messagebox.showerror("Error", f"Camera error: {str(e)}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showwarning("Warning", "Could not read frame from camera")
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_frame, 1.1, 4)
        for (x, y, w, h) in faces:
            face_roi = gray_frame[y:y+h, x:x+w]
            try:
                label, confidence = face_recognizer_model.predict(face_roi)
                name = label_names.get(label, f"Unknown (ID:{label})")
                display_text = f"{name}" if confidence <= 85 else f"Unknown ({confidence:.1f})"
                color = (0, 255, 0) if confidence <= 85 else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, display_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            except Exception as e:
                print(f"Recognition error: {str(e)}")
        cv2.imshow("Face Recognition - Press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

def main():
    global root
    root = Tk()
    root.geometry("300x200")
    root.resizable(False, False)
    root.title("Face Recognition App")
    create_main_frame()
    root.mainloop()

main()
