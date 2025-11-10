import cv2
import face_recognition
import os
import csv
from datetime import datetime

# ------------------ SETTINGS ------------------
print("\n🎓 SMART ATTENDANCE SYSTEM (Configurable Mode)")
min_duration = int(input("Enter minimum duration (in minutes) for Present (default 30): ") or 30)
EXIT_DELAY = int(input("Enter minimum delay (in seconds) between entry and exit scans (default 30): ") or 30)
print(f"\n🕒 Minimum duration for Present: {min_duration} min")
print(f"⏳ Exit scan delay: {EXIT_DELAY} sec")
print("➡️ Scan once when entering and once when leaving.\n")

# ------------------ LOAD KNOWN FACES ------------------
path = "students"
known_encodings = []
known_names = []

for filename in os.listdir(path):
    if filename.endswith((".jpg", ".png")):
        img = face_recognition.load_image_file(os.path.join(path, filename))
        encodings = face_recognition.face_encodings(img)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(filename)[0])

# Attendance storage
attendance_file = "attendance.csv"
student_records = {}  # {name: {"entry": time, "exit": time}}

# ------------------ CAMERA START ------------------
video = cv2.VideoCapture(0)
last_message = ""
message_time = datetime.now()

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

            # Entry Scan
            if name not in student_records or student_records[name].get("entry") is None:
                student_records[name] = {"entry": current_time, "exit": None}
                last_message = f"✅ Entry recorded for {name}"
                message_time = current_time
                print(f"[ENTRY] {name} at {current_time.strftime('%H:%M:%S')}")

            # Exit Scan (only after EXIT_DELAY)
            elif student_records[name].get("exit") is None:
                entry_time = student_records[name]["entry"]
                time_since_entry = (current_time - entry_time).total_seconds()

                if time_since_entry > EXIT_DELAY:
                    student_records[name]["exit"] = current_time
                    duration = (student_records[name]["exit"] - entry_time).total_seconds() / 60
                    status = "Present" if duration >= min_duration else "Absent"

                    last_message = (
                        f"✅ Exit recorded for {name} ({status})"
                        if status == "Present"
                        else f"❌ Exit recorded for {name} ({status})"
                    )
                    message_time = current_time

                    print(f"[EXIT] {name} at {current_time.strftime('%H:%M:%S')} | Duration: {duration:.2f} min | {status}")

                    # Save to CSV
                    with open(attendance_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([name, entry_time, student_records[name]["exit"], f"{duration:.2f} min", status])

                else:
                    print(f"[INFO] {name} scanned too soon for exit ({time_since_entry:.1f}s since entry)")

            # Draw detection box
            top, right, bottom, left = loc
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display message
    if last_message and (datetime.now() - message_time).total_seconds() < 3:
        cv2.putText(frame, last_message, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Smart Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
print("\n✅ Attendance session ended.")
