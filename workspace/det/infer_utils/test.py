import sys
import coremltools  
from PIL import Image  
import numpy as np
import cv2
import requests
from io import BytesIO

model = coremltools.models.MLModel('./deploy_graph.mlmodel')
INPUT_SIZE = 512
imgpath = sys.argv[1]

#img_url = 'https://upload.wikimedia.org/wikipedia/commons/9/93/Golden_Retriever_Carlos_%2810581910556%29.jpg'
#response = requests.get(img_url)
#img = Image.open(BytesIO(response.content))
#img = img.resize([512,512], Image.ANTIALIAS)

# read image
img = Image.open(imgpath)  

# calculate size
width, height = img.size
large_one = max(width, height)
scale = float(INPUT_SIZE) / float(large_one)

new_width = 0
new_height = 0
if width >= height:
  new_width = INPUT_SIZE
  new_height = int(height * scale)
else:
  new_height = INPUT_SIZE
  new_width = int(width * scale)

# resize
img = img.resize((new_width, new_height), Image.ANTIALIAS)

# padding
delta_w = INPUT_SIZE - new_width
delta_h = INPUT_SIZE - new_height
top, bottom = 0, delta_h
left, right = 0, delta_w
color = [127, 127, 127]
img_array = np.array(img)
img_array = cv2.copyMakeBorder(img_array, top, bottom, left, right, cv2.BORDER_CONSTANT,
    value=color)
img = Image.fromarray(np.uint8(img_array))

# run the model
coreml_inputs = {'image__0': img}
res = model.predict(coreml_inputs, useCPUOnly=True) 


# add softmax
logits = res['MobilenetV2__Logits__Conv2d_1c_1x1__BiasAdd__0']
upsampled = res['MobilenetV2__ResizeBilinear__0']
#print(logits.shape)
#add_softmax(name, input_name, output_name)

#spec = model.get_spec()
#model.save('another.mlmodel')
#coremltools.models.utils.save_spec(spec, 'another.mlmodel')
#coremltools.models.utils.visualize_spec(spec, port=8989)



# post process
resized = res['MobilenetV2__ResizeBilinear__0']
resized = np.squeeze(resized)
mask_array = np.argmax(resized, axis=0)

mask = Image.fromarray(np.uint8(mask_array))
mask_pixel = mask.load()
embed = img
embed_pix = embed.load()
for i in range(INPUT_SIZE):
  for j in range(INPUT_SIZE):
    if mask_pixel[i,j] == 1:
      mask.putpixel((i, j), 255)
      r, g, b = embed_pix[i, j]
      embed_pix[i, j] = (r+100, g, b+100)
    else:
      r, g, b = embed_pix[i, j]
      embed_pix[i, j] = (r, g/2, b/2)

embed_array = np.array(embed)
embed_crop_array = embed_array[0:new_height, 0:new_width]
embed_crop = Image.fromarray(np.uint8(embed_crop_array))
embed_crop.save('embed.png')

mask_array = np.array(mask)
mask_crop_array = mask_array[0:new_height, 0:new_width]
mask_crop = Image.fromarray(np.uint8(mask_crop_array))
mask_crop.save('mask.png')



