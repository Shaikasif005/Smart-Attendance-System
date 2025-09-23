# attendance_system.py
import os
import cv2
import pickle
import csv
import time
import sys
from datetime import datetime

try:
    import face_recognition
    import numpy as np
except Exception as e:
    print("Missing required modules. Install with:\n"
          "pip install opencv-python face-recognition face-recognition-models numpy\n"
          "(If dlib fails to build use dlib-bin on Windows: pip install dlib-bin)\n")
    raise

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STUDENTS_DIR = os.path.join(BASE_DIR, "students")
ENCODINGS_FILE = os.path.join(BASE_DIR, "encodings.pkl")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")

os.makedirs(STUDENTS_DIR, exist_ok=True)


def save_encodings(data):
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)


def load_encodings():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {"names": [], "encodings": []}


def build_encodings_from_images():
    """Scan students/ for either:
       - subfolders (students/<name>/*.jpg) OR
       - images in students/ with filenames as names (alice.jpg)
       and build encodings list.
    """
    names = []
    encodings = []

    # Case 1: subfolders per student
    for entry in os.listdir(STUDENTS_DIR):
        p = os.path.join(STUDENTS_DIR, entry)
        if os.path.isdir(p):
            for imgf in os.listdir(p):
                if imgf.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(p, imgf)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    enc = face_recognition.face_encodings(rgb)
                    if enc:
                        encodings.append(enc[0])
                        names.append(entry)

    # Case 2: images directly in students/ (filename -> name)
    for entry in os.listdir(STUDENTS_DIR):
        p = os.path.join(STUDENTS_DIR, entry)
        if os.path.isfile(p) and entry.lower().endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(entry)[0]
            img = cv2.imread(p)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            enc = face_recognition.face_encodings(rgb)
            if enc:
                encodings.append(enc[0])
                names.append(name)

    data = {"names": names, "encodings": encodings}
    save_encodings(data)
    return data


def ensure_attendance_file():
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])


def enroll_student_interactive(name=None, capture_count=8):
    """Enroll by capturing images with webcam. Saves into students/<name>/"""
    if name is None:
        name = input("Enter student name (no spaces recommended): ").strip()
        if not name:
            print("Invalid name.")
            return

    student_dir = os.path.join(STUDENTS_DIR, name)
    os.makedirs(student_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("\nEnrollment mode:")
    print(" - Press 'c' to capture an image when ready.")
    print(" - Capture multiple angles: look straight, left, right, up, down.")
    print(" - Press 'q' to quit early.\n")

    count = 0
    while count < capture_count:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        cv2.putText(frame, f"Name: {name}  Captures: {count}/{capture_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        cv2.putText(frame, "Press 'c' to capture, 'q' to quit.",
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow("Enroll - Press 'c' to capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            img_path = os.path.join(student_dir, f"{name}_{count+1}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Captured: {img_path}")
            count += 1
            # small pause so user can move slightly
            time.sleep(0.5)
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Build encodings after capture
    print("Encoding captured images...")
    data = build_encodings_from_images()
    print(f"Done. Total registered encodings: {len(data['names'])}")


def recognize_and_mark(tolerance=0.5, scale=0.25):
    data = load_encodings()
    if len(data["encodings"]) == 0:
        print("No registered encodings found. Building from `students/` folder now...")
        data = build_encodings_from_images()

    if len(data["encodings"]) == 0:
        print("No students registered yet. Use option 1 to enroll students (or drop images into students/).")
        return

    print(f"Loaded {len(data['names'])} encodings. Starting webcam for recognition...")
    ensure_attendance_file()

    # in-session set to avoid duplicate entries in same run
    session_marked = set()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # speed up by shrinking image for detection
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        encodings_current = face_recognition.face_encodings(rgb_small, face_locations)

        for enc, loc in zip(encodings_current, face_locations):
            face_distances = face_recognition.face_distance(data["encodings"], enc)
            if len(face_distances) == 0:
                continue

            best_idx = int(np.argmin(face_distances))
            best_dist = face_distances[best_idx]

            if best_dist <= tolerance:
                name = data["names"][best_idx]
                # prevent duplicates in same session
                if name not in session_marked:
                    now = datetime.now()
                    date_str = now.strftime("%Y-%m-%d")
                    time_str = now.strftime("%H:%M:%S")
                    with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([name, date_str, time_str])
                    session_marked.add(name)
                    print(f"[{date_str} {time_str}] Marked attendance: {name}")

                # draw box on original frame (scale coords back up)
                top, right, bottom, left = loc
                top *= int(1/scale); right *= int(1/scale)
                bottom *= int(1/scale); left *= int(1/scale)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                # unknown
                top, right, bottom, left = loc
                top *= int(1/scale); right *= int(1/scale)
                bottom *= int(1/scale); left *= int(1/scale)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (left, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Attendance (press 'e' to enroll, 'q' to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('e'):
            # quick switch to enroll (ask name in console)
            cap.release()
            cv2.destroyAllWindows()
            name = input("Enter name to enroll: ").strip()
            if name:
                enroll_student_interactive(name=name)
            # reload encodings after enrollment
            data = load_encodings()
            cap = cv2.VideoCapture(0)

    cap.release()
    cv2.destroyAllWindows()
    print("Session ended. Attendance file:", ATTENDANCE_FILE)


def main_menu():
    print("Smart Attendance System (webcam-only)")
    print("Project folder:", BASE_DIR)
    print("Students folder:", STUDENTS_DIR)
    print("Encodings file:", ENCODINGS_FILE)
    print("Attendance file:", ATTENDANCE_FILE)
    print("\nOptions:")
    print("1) Enroll new student (use webcam to capture multiple angles)")
    print("2) Start attendance (recognize & save to CSV)")
    print("3) Build/refresh encodings from existing images in students/")
    print("4) Exit\n")

    choice = input("Choose (1/2/3/4): ").strip()
    if choice == "1":
        enroll_student_interactive()
    elif choice == "2":
        recognize_and_mark()
    elif choice == "3":
        print("Building encodings from images...")
        data = build_encodings_from_images()
        print(f"Built {len(data['encodings'])} encodings for {len(set(data['names']))} names.")
    else:
        print("Exiting.")
        sys.exit(0)


if __name__ == "__main__":
    while True:
        main_menu()
