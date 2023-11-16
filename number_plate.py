import cv2

# Path to the pre-trained Haar Cascade classifier for Russian license plates
harcascade = "model/haarcascade_russian_plate_number.xml"

# Open a video capture object (0 indicates the default camera)
cap = cv2.VideoCapture(0)

# Set the width and height of the video capture
cap.set(3, 640)  # width
cap.set(4, 480)  # height

# Minimum area for a detected region to be considered a license plate
min_area = 500

# Counter to keep track of saved plate images
count = 0

# Main loop for capturing and processing frames from the camera
while True:
    # Read a frame from the video capture
    success, img = cap.read()

    # Create a CascadeClassifier object using the pre-trained classifier
    plate_cascade = cv2.CascadeClassifier(harcascade)

    # Convert the frame to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect license plates in the grayscale image
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    # Loop through the detected plates and process each one
    for (x, y, w, h) in plates:
        # Calculate the area of the detected region
        area = w * h

        # Check if the area is greater than the minimum area
        if area > min_area:
            # Draw a rectangle around the detected plate
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Add text label for the detected plate
            cv2.putText(img, "Number Plate", (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            # Extract the region of interest (ROI) containing the plate
            img_roi = img[y: y+h, x:x+w]

            # Display the ROI in a separate window
            cv2.imshow("ROI", img_roi)

    # Display the original image with detected plates
    cv2.imshow("Result", img)

    # Check if the 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        # Save the captured plate image with a unique filename
        cv2.imwrite("plates/scaned_img_" + str(count) + ".jpg", img_roi)

        # Display a confirmation message on the original image
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)

        # Display the modified image with the confirmation message
        cv2.imshow("Results", img)

        # Wait for a short duration (500 milliseconds)
        cv2.waitKey(500)

        # Increment the counter for saved plate images
        count += 1
