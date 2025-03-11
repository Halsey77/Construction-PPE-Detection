import cv2
import numpy as np
from ultralytics import YOLO


class PPEApp:
    def __init__(self):
        # Initialize the YOLO model
        self.model = YOLO(
            "Model/yolov11-n-construct.pt"
        )  # Replace with your model path
        self.confidence_threshold = 0.8

        # Initialize webcam
        self.cap = cv2.VideoCapture("test_videos/night-construct-1.mp4")  # 0 is usually the default camera
        if not self.cap.isOpened():
            print("Error: Unable to access the webcam.")
            exit()

        self.colors = [
            (255, 0, 0),  # Hardhat (Blue)
            (0, 255, 0),  # Mask (Green)
            (0, 0, 255),  # NO-Hardhat (Red)
            (255, 255, 0),  # NO-Mask (Cyan)
            (255, 0, 255),  # NO-Safety Vest (Magenta)
            (0, 255, 255),  # Person (Yellow)
            (128, 0, 128),  # Safety Cone (Purple)
            (128, 128, 0),  # Safety Vest (Olive)
            (0, 128, 128),  # Machinery (Teal)
            (128, 128, 128),  # Vehicle (Gray)
        ]
        
        self.window_name = "yolov11 nano feed"

        # Create a resizable window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def draw_text_with_background(
        self,
        frame,
        text,
        position,
        font_scale=0.4,
        color=(255, 255, 255),
        thickness=2,
        bg_color=(0, 0, 0),
        alpha=0.7,
        padding=5,
    ):
        font = cv2.FONT_HERSHEY_TRIPLEX
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_width, text_height = text_size
        overlay = frame.copy()
        x, y = position
        cv2.rectangle(
            overlay,
            (x - padding, y - text_height - padding),
            (x + text_width + padding, y + padding),
            bg_color,
            -1,
        )
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture frame.")
            return

        # Initialize counters
        hardhat_count = 0
        vest_count = 0
        person_count = 0

        # Perform YOLO inference with dynamic confidence
        # results = self.model(frame, conf=self.confidence_threshold, iou=0.3)
        results = self.model(frame)

        # Loop through the results and draw bounding boxes
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    confidence = box.conf[0]  # Confidence score
                    cls = int(box.cls[0])  # Class ID
                    label = f"{self.model.names[cls]} ({confidence:.2f})"

                    color = self.colors[cls % len(self.colors)]

                    # Draw the bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    self.draw_text_with_background(
                        frame,
                        label,
                        (x1, y1 - 10),
                        font_scale=1,
                        color=(255, 255, 255),
                        bg_color=color,
                        alpha=0.8,
                        padding=4,
                    )

                    if self.model.names[cls] == "Hardhat":
                        hardhat_count += 1
                    elif self.model.names[cls] == "Safety Vest":
                        vest_count += 1
                    elif self.model.names[cls] == "Person":
                        person_count += 1

        # Add the counts on the sideboard
        sideboard_text = [
            f"Hardhats: {hardhat_count}",
            f"Safety Vests: {vest_count}",
            f"People: {person_count}",
        ]

        y_position = 30
        for text in sideboard_text:
            self.draw_text_with_background(
                frame,
                text,
                (10, y_position),
                font_scale=1,
                color=(255, 255, 255),
                bg_color=(0, 0, 0),
                alpha=0.7,
                padding=5,
            )
            y_position += 30

        # Resize the frame to fit the window dynamically
        resized_frame = cv2.resize(
            frame, (640, 480), interpolation=cv2.INTER_LINEAR
        )  # Resize to a fixed size

        # Display the annotated frame
        cv2.imshow(self.window_name, resized_frame)

    def run(self):
        while True:
            self.update_frame()
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Release the webcam and close any open windows
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = PPEApp()
    app.run()
