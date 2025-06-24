# from main_simple_lib import *

from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pdb

# im = load_image('https://viper.cs.columbia.edu/static/images/kids_muffins.jpg')
# query = 'How many muffins can each kid have for it to be fair?'


# im = load_image('/home/jovyan/workspace/datasets/MNMath_Add_3digit/val/0.png')
# query = "## Image description:\nThe image shows two rows of handwritten digits. Each row represents a 3-digit number.\n## Rule:\n$Y$ is the sum of these two numbers.\n## Query:\nWhat is $Y$?"
# query = "## Image description:\nThe image shows 6 handwritten digits. The first 3 digits on the top represents a 3-digit number, and the second 3 digits on the bottom represents another 3-digit number.\n## Rule:\n$Y$ is the sum of these two numbers.\n## Query:\nWhat is $Y$?"
# query = "## Image description:\nThe image shows two rows of handwritten digits. Each row represents a 3-digit number.\n## Rule:\n$Y$ is the sum of these two numbers.\n## Query:\nWhat is $Y$?"

# im = load_image('/home/jovyan/workspace/datasets/MNLogic_XOR_3digit/test/0.png')
# query = "## Image description:\nThe image shows 3 handwritten binary digits.\n## Rule:\n$Y$ is the XOR of these binary digits.\n## Query:\nWhat is $Y$?"

# im = load_image('/home/jovyan/workspace/datasets/KandLogic_3obj/val/0.png')
# # query = "## Image description:\nThe image shows 3 geometric objects, each with a specific shape (square, triangle, circle) and color (red, blue, yellow).\n## Rule:\nIf all objects with the same shape have the same color, then $Y$ is True. Otherwise, $Y$ is False.\n## Query:\nWhat is $Y$?"
# query = "## Image description:\nThe image shows 3 geometric objects.\n## Rule:\nIf all objects with the same shape have the same color, then $Y$ is True. Otherwise, $Y$ is False.\n## Query:\nWhat is $Y$?"

# # show_single_image(im)
# code = get_code(query)
# print(code[0])
# execute_code(code, im, show_intermediate_steps=True)



# Load image
img_path = "/home/jovyan/workspace/datasets/MNMath_Add_3digit/val/5.png"
img = Image.open(img_path).convert("RGB")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model_id = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
model.eval()
model.requires_grad_(False)
model.to(device)

# Define text
text = ['digit']
text = [t.lower().strip().rstrip('.') + '.' for t in text]

# Object detection
inputs = processor(images=img, text=" ".join(text), return_tensors="pt").to(device)
outputs = model(**inputs)

target_sizes = [img.size[::-1]]  # (height, width)
results = processor.post_process_grounded_object_detection(
    outputs=outputs,
    input_ids=inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=target_sizes
)

boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]

# Format boxes to [left, lower, right, upper] with flipped Y-axis to match your convention
# left, upper, right, lower = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
left, lower, right, upper = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] #
height = img.size[1]
# boxes = torch.stack([left, height - lower, right, height - upper], -1)
boxes = torch.stack([left, lower, right, upper], -1) #
boxes = boxes.cpu()

# Visualize
# Convert image to tensor for plotting
img_tensor = TF.to_tensor(img).permute(1, 2, 0).numpy()

# Plot image
fig, ax = plt.subplots(1)
ax.imshow(img_tensor)

# Draw bounding boxes
for box, score, label in zip(boxes, scores, labels):
    x0, y0, x1, y1 = box.tolist() # left, lower, right, upper
    width, height = x1 - x0, y1 - y0
    rect = patches.Rectangle(
        (x0, y0), width, height,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)
    # ax.text(
    #     x0, y0 - 5,
    #     f"{text[label]}: {score:.2f}",
    #     color='white', backgroundcolor='red', fontsize=8
    # )

plt.axis('off')
plt.tight_layout()
plt.savefig("/home/jovyan/workspace/viper/tmp.png", bbox_inches='tight')
plt.show()