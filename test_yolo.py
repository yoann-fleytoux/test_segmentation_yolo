from ultralytics import YOLO
import numpy as np
import cv2
import random
import torch
import os

# opening the file in read mode
my_file = open("coco.txt", "r")
# reading the file
data = my_file.read()
# replacing end splitting the text | when newline ('\n') is seen.
class_list = data.split("\n")
my_file.close()

# print(class_list)

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    detection_colors.append((b,g,r))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using Device: ", device)


# load a pretrained YOLOv8n model
#model = YOLO("yolov8m.pt", "v8")  
# Load a pretrained YOLOv8n-seg model
model = YOLO("yolov8n-seg.pt")

image_path = "clutter_1467.png"
frame = cv2.imread(image_path)
#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Set the minimum and maximum area thresholds (in pixels)
min_area = 100
max_area = 10000

mask_directory = "binary_masks"
if not os.path.exists(mask_directory):
    os.makedirs(mask_directory)

# predict on an image
detect_params = model.predict(source=[frame], conf=0.01, save=False) 

#frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

font = cv2.FONT_HERSHEY_COMPLEX

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

detection_colors = [
    (255, 0, 0)      ,# red
    (0, 255, 0)      ,# green
    (0, 0, 255)      ,# blue
    (255, 255, 0)    ,# yellow
    (0, 255, 255)    ,# cyan
    (255, 0, 255)    ,# magenta
    (128, 0, 0)      ,# maroon
    (128, 0, 128)    ,# purple
    (0, 128, 0)      ,# green (dark)
    (0, 0, 128)      ,# blue (dark)
    (128, 128, 0)    ,# olive
    (128, 128, 128)  ,# gray
    (192, 192, 192)  ,# silver
    (255, 255, 255)  ,# white
    (255, 99, 71)    ,# tomato
    (255, 69, 0)     ,# orange red
    (255, 140, 0)    ,# dark orange
    (255, 215, 0)    ,# gold
    (218, 165, 32)   ,# goldenrod
    (154, 205, 50)   ,# yellow green
    (50, 205, 50)    ,# lime green
    (0, 128, 0)      ,# green (dark)
    (34, 139, 34)    ,# forest green
    (0, 100, 0)      ,# green (dark)
    (0, 255, 0)      ,# green
    (0, 255, 127)    ,# spring green
    (46, 139, 87)    ,# sea green
    (60, 179, 113)   ,# medium sea green
    (143, 188, 143)  ,# dark sea green
    (255, 0, 255)    ,# magenta
    (218, 112, 214)  ,# orchid
    (199, 21, 133)   ,# medium violet red
    (219, 112, 147)  ,# pale violet red
    (255, 20, 147)   ,# deep pink
    (255, 192, 203)  ,# pink
    (255, 182, 193)  ,# light pink
    (139, 0, 139)    ,# dark magenta
    (128, 0, 0)      ,# maroon
    (139, 69, 19)    ,# saddle brown
    (160, 82, 45)    ,# sienna
    (165, 42, 42)    ,# brown
    (210, 105, 30)   ,# chocolate
    (244, 164, 96)   ,# sandy brown
    (222, 184, 135)  ,# burlywood
    (250, 128, 114)  ,# salmon
    (233, 150, 122)  ,# dark salmon
    (255, 160, 122)  ,# light salmon
    (255, 165, 0)    ,# orange
    (255, 228, 181)  ,# moccasin
    (255, 235, 205)  ,# blanched almond
    (255, 248, 220)  ,# cornsilk
    (255, 250, 205) ,# lemon chiffon
    (255, 255, 224) ,# light yellow
    (255, 255, 240) ,# ivory
    (250, 250, 210) ,# light goldenrod yellow
    (255, 215, 0) ,# gold
    (255, 228, 196) ,# bisque
    (245, 222, 179) ,# wheat
    (222, 184, 135) ,# burlywood
    (210, 180, 140) ,# tan
    (188, 143, 143) ,# rosy brown
    (169, 169, 169) ,# dark gray
    (128, 128, 128) ,# gray
    (105, 105, 105) ,# dim gray
    (119, 136, 153) ,# light slate gray
    (112, 128, 144) ,# slate gray
    (47, 79, 79) ,# dark slate gray
    (0, 0, 128) ,# navy
    (25, 25, 112) ,# midnight blue
    (0, 0, 205) ,# medium blue
    (65, 105, 225) ,# royal blue
    (30, 144, 255) ,# dodger blue
    (0, 191, 255) ,# deep sky blue
    (135, 206, 250) ,# light sky blue
    (70, 130, 180) ,# steel blue
    (176, 196, 222) ,# light steel blue
    (173, 216, 230) ,# light blue
    (240, 248, 255) ,# alice blue
    (240, 255, 255) ,# azure
    (248, 248, 255) ,# ghost white
    (255, 228, 225) ,# misty rose
    (255, 99, 71) ,# tomato
    (255, 127, 80) ,# coral
    (255, 160, 122) ,# light coral
    (255, 140, 0) ,# dark orange
    (255, 165, 0) ,# orange
    (255, 215, 0) ,# gold
    (218, 165, 32) ,# goldenrod
    (128, 128, 0) ,# olive
    (128, 0, 0) ,# maroon
    (255, 255, 255) ,# white
    (245, 245, 245) ,# white smoke
    (220, 220, 220) ,# gainsboro
    (211, 211, 211) ,# light gray
    (192, 192, 192) ,# silver
    (169, 169, 169) ,# dark gray
    (128, 128, 128) ,# gray
    (105, 105, 105) ,# dim gray
    (0, 0, 0) ,# black
]
random.shuffle(detection_colors)

# Create a copy of the original frame to draw masks on
frame_with_masks = frame.copy()

# Get the masks as a numpy array
# masks_np = detect_params[0].masks.cpu().numpy()

# Get the masks as a numpy array
masks = detect_params[0].masks

# Create an empty list to keep track of the indices of the selected masks
selected_masks = []

# Iterate through the detected instances in reverse order of confidence
for i in reversed(np.argsort(detect_params[0].boxes.conf.cpu().numpy())):
    # Get the bounding box of the mask
    bb = detect_params[0].boxes[i].cpu().xyxy.numpy()[0]

    # Check if the bounding box intersects with any of the selected masks
    intersects = False
    for j in selected_masks:
        bb_selected = detect_params[0].boxes[j].cpu().xyxy.numpy()[0]
        if bb[0] < bb_selected[2] and bb[2] > bb_selected[0] and bb[1] < bb_selected[3] and bb[3] > bb_selected[1]:
            intersects = True
            
            # Find the index of the mask with the highest confidence among the intersecting masks
            intersecting_indices = [j for j in selected_masks if detect_params[0].boxes[j].cls.cpu().numpy()[0] == detect_params[0].boxes[i].cls.cpu().numpy()[0]]
            intersecting_confidences = [detect_params[0].boxes[j].conf.cpu().numpy()[0] for j in intersecting_indices] + [detect_params[0].boxes[i].conf.cpu().numpy()[0]]
            if intersecting_indices:
                max_index = intersecting_indices[np.argmax(intersecting_confidences)]
                
                # Replace the mask with the lowest confidence with the one with the highest confidence
                if detect_params[0].boxes[i].conf.cpu().numpy()[0] > detect_params[0].boxes[max_index].conf.cpu().numpy()[0]:
                    selected_masks.remove(max_index)
                    selected_masks.append(i)
            else:
                selected_masks.append(i)
                
            break

    if not intersects:
        selected_masks.append(i)
    else:
        # Find the index of the mask with the highest confidence among the intersecting masks
        intersecting_indices = [j for j in selected_masks if detect_params[0].boxes[j].cls.cpu().numpy()[0] == detect_params[0].boxes[i].cls.cpu().numpy()[0]]
        intersecting_confidences = [detect_params[0].boxes[j].conf.cpu().numpy()[0] for j in intersecting_indices] + [detect_params[0].boxes[i].conf.cpu().numpy()[0]]
        if intersecting_indices:
            max_index = intersecting_indices[np.argmax(intersecting_confidences)]
            
            # Replace the mask with the lowest confidence with the one with the highest confidence
            if detect_params[0].boxes[i].conf.cpu().numpy()[0] > detect_params[0].boxes[max_index].conf.cpu().numpy()[0]:
                selected_masks.remove(max_index)
                selected_masks.append(i)
        else:
            selected_masks.append(i)


# Iterate through the detected instances
#for i in range(masks.shape[0]):
# Iterate through the selected masks and process them
for i in selected_masks:
    # Get the mask, class ID, and color
    mask = (masks.data[i].cpu().numpy() * 255).astype(np.uint8)
    area = cv2.countNonZero(mask)
    # Check if the mask meets the area criteria
    if area < min_area or area > max_area:
        continue  # skip this mask

    clsID = detect_params[0].boxes[i].cls.cpu().numpy()[0]
    clsName = class_list[int(clsID)]
    color = detection_colors[int(clsID)]
    conf = detect_params[0].boxes[i].conf.cpu().numpy()[0]

    # Resize the mask to match the original image size
    resized_mask = cv2.resize(mask, (frame_with_masks.shape[1], frame_with_masks.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Generate a unique filename for each mask
    filename = os.path.join(mask_directory, f"{clsName}_{i}.png")

    # Save the binary mask as a PNG image
    cv2.imwrite(filename, resized_mask)

    # Colorize the mask
    #colored_mask = cv2.merge([resized_mask, resized_mask, resized_mask]) * np.array(color)
    colored_mask = np.zeros_like(frame_with_masks)
    colored_mask[:, :, 0] = resized_mask * color[0]
    colored_mask[:, :, 1] = resized_mask * color[1]
    colored_mask[:, :, 2] = resized_mask * color[2]

    # Add class name to the mask
    # Display class name and confidence
    bb = detect_params[0].boxes[i].cpu().xyxy.numpy()[0]

    cv2.putText(
        frame_with_masks,
        class_list[int(clsID)]
        + " "
        + str(round(conf, 3))
        + "%",
        (int(bb[0]), int(bb[1]) - 10),
        font,
        1,
        (255, 255, 255),
        2,
    )
    # Superpose the colored mask on the original image
    #frame_with_masks = cv2.addWeighted(frame_with_masks, 1, colored_mask, 0.5, 0)
    frame_with_masks = cv2.addWeighted(frame_with_masks.astype(np.float32), 1, colored_mask.astype(np.float32), 0.5, 0, dtype=cv2.CV_32F)
# Save the image with superposed masks
cv2.imwrite("image_with_masks.png", frame_with_masks)
# else:
# Predict on image
#detect_params = model.predict(source=[frame], conf=0.45, save=False)

# Convert tensor array to numpy
DP = detect_params[0].cpu().numpy()
print(DP)

if len(DP) != 0:
    frame_with_masks = frame.copy()

    for i in range(len(detect_params[0])):
        print(i)

        boxes = detect_params[0].boxes
        box = boxes[i].cpu()  # returns one box
        clsID = box.cls.numpy()[0]
        conf = box.conf.numpy()[0]
        bb = box.xyxy.numpy()[0]

        cv2.rectangle(
            frame,
            (int(bb[0]), int(bb[1])),
            (int(bb[2]), int(bb[3])),
            detection_colors[int(clsID)],
            3,
        )

        # Display class name and confidence
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(
            frame,
            class_list[int(clsID)]
            + " "
            + str(round(conf, 3))
            + "%",
            (int(bb[0]), int(bb[1]) - 10),
            font,
            1,
            (255, 255, 255),
            2,
        )

        # Create a binary mask for the detected object
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        cv2.rectangle(
            mask,
            (int(bb[0]), int(bb[1])),
            (int(bb[2]), int(bb[3])),
            255,  # White color for the object
            -1,  # Filled rectangle
        )

        # Save the binary mask in the specified directory
        mask_filename = os.path.join(mask_directory, f"mask_{i}.png")
        cv2.imwrite(mask_filename, mask)


    cv2.imwrite('image_with_rectangles.png', frame)
