import sys
import coremltools  
from PIL import Image  
import numpy as np
import cv2
import requests
from io import BytesIO

model = coremltools.models.MLModel('./det/deploy_graph.mlmodel')
INPUT_SIZE = 300
imgpath = sys.argv[1]

#img_url = 'https://upload.wikimedia.org/wikipedia/commons/9/93/Golden_Retriever_Carlos_%2810581910556%29.jpg'
#response = requests.get(img_url)
#img = Image.open(BytesIO(response.content))
#img = img.resize([512,512], Image.ANTIALIAS)

# read image
img = Image.open(imgpath)
img = img.resize([300, 300], Image.BILINEAR) 

# run the model
coreml_inputs = {'image__0': img}
res = model.predict(coreml_inputs, useCPUOnly=True) 

box_encodings = res['box_encodings__0']
class_scores = res['class_scores__0']

print('box_encodings.shape: ', box_encodings.shape)
print('class_scores.shape: ', class_scores.shape)

sorted_scores = np.sort(class_scores[1,:,0])
print('Top 20 person scores: ', sorted_scores[class_scores.shape[1] - 20:])

