from main_simple_lib import *
import pdb

# im = load_image('https://viper.cs.columbia.edu/static/images/kids_muffins.jpg')
# query = 'How many muffins can each kid have for it to be fair?'

im = load_image('/home/jovyan/workspace/datasets/MNMath_Add_3digit/val/0.png')
# query = "## Image description:\nThe image shows two rows of handwritten digits. Each row represents a 3-digit number.\n## Rule:\n$Y$ is the sum of these two numbers.\n## Query:\nWhat is $Y$?"
query = "## Image description:\nThe image shows 6 handwritten digits. The first 3 digits on the top represents a 3-digit number, and the second 3 digits on the bottom represents another 3-digit number.\n## Rule:\n$Y$ is the sum of these two numbers.\n## Query:\nWhat is $Y$?"

# im = load_image('/home/jovyan/workspace/datasets/MNLogic_XOR_3digit/val/0.png')
# query = "## Image description:\nThe image shows 3 handwritten binary digits.\n## Rule:\n$Y$ is the XOR of these binary digits.\n## Query:\nWhat is $Y$?"

# im = load_image('/home/jovyan/workspace/datasets/KandLogic_3obj/val/0.png')
# # query = "## Image description:\nThe image shows 3 geometric objects, each with a specific shape (square, triangle, circle) and color (red, blue, yellow).\n## Rule:\nIf all objects with the same shape have the same color, then $Y$ is True. Otherwise, $Y$ is False.\n## Query:\nWhat is $Y$?"
# query = "## Image description:\nThe image shows 3 geometric objects.\n## Rule:\nIf all objects with the same shape have the same color, then $Y$ is True. Otherwise, $Y$ is False.\n## Query:\nWhat is $Y$?"

# show_single_image(im)
code = get_code(query)
execute_code(code, im, show_intermediate_steps=True)
