import cv2
import face_recognition
import os
import csv
from datetime import datetime

# Track entry & exit times
student_times = {}  # {student_name: {"entry": time, "last_seen": time, "exit": None}}
attendance_file = "attendance.csv"

# Load known faces
path = "students"
known_encodings = []
known_names = []

for filename in os.listdir(path):
    if filename.endswith((".jpg", ".png")):
        img = face_recognition.load_image_file(os.path.join(path, filename))
        encoding = face_recognition.face_encodings(img)[0]
        known_encodings.append(encoding)
        known_names.append(os.path.splitext(filename)[0])

# Start camera
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb_frame)
    encodings = face_recognition.face_encodings(rgb_frame, locations)

    current_time = datetime.now()

    for encoding, loc in zip(encodings, locations):
        matches = face_recognition.compare_faces(known_encodings, encoding)
        name = "Unknown"

        if True in matches:
            name = known_names[matches.index(True)]

            # If first time detected → entry
            if name not in student_times:
                student_times[name] = {"entry": current_time, "last_seen": current_time, "exit": None}
                print(f"[ENTRY] {name} at {current_time}")

            else:
                # Update last seen
                student_times[name]["last_seen"] = current_time

            top, right, bottom, left = loc
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Check for students who disappeared (not seen for 30 sec)
    for student, times in list(student_times.items()):
        if times["exit"] is None:
            diff = (current_time - times["last_seen"]).total_seconds()
            if diff > 30:  # 30 sec grace period
                times["exit"] = current_time
                duration = (times["exit"] - times["entry"]).total_seconds() / 60

                status = "Present" if duration >= 30 else "Absent"
                print(f"[EXIT] {student} at {current_time} | Duration: {duration:.2f} min | {status}")

                # Save to CSV
                with open(attendance_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([student, times["entry"], times["exit"], f"{duration:.2f} min", status])

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
