import coremltools  
from PIL import Image  
import numpy as np
import requests
from io import BytesIO


model = coremltools.models.MLModel('./mnetv2cls.mlmodel')
imgsize = 224

#img_url = 'http://pic.sc.chinaz.com/files/pic/pic9/201407/apic4616.jpg'
#response = requests.get(img_url)
#img = Image.open(BytesIO(response.content))
#img = img.resize([imgsize,imgsize], Image.ANTIALIAS)

#img = Image.open('flowers/dan/10443973_aeb97513fc_m.jpg')
#img = Image.open('flowers/dan/10437652486_aa86c14985.jpg')
#img = Image.open('flowers/dan/10477378514_9ffbcec4cf_m.jpg')
#img = Image.open('flowers/dan/10486992895_20b344ce2d_n.jpg')

#img = Image.open('flowers/daisy/10555749515_13a12a026e.jpg')
#img = Image.open('flowers/daisy/10555815624_dc211569b0.jpg')
#img = Image.open('flowers/daisy/10555826524_423eb8bf71_n.jpg')

#img = Image.open('flowers/rose/102501987_3cdb8e5394_n.jpg')
#img = Image.open('flowers/rose/10090824183_d02c613f10_m.jpg')
img = Image.open('flowers/rose/10503217854_e66a804309.jpg')
#img = Image.open('flowers/rose/10894627425_ec76bbc757_n.jpg')

img = img.resize([imgsize,imgsize], Image.ANTIALIAS)

#coreml_inputs = {'MobilenetV2__input__0': img}
coreml_inputs = {'input__0': img}

res = model.predict(coreml_inputs, useCPUOnly=True) 

logits = res['MobilenetV2__Predictions__Reshape_1__0']
print(logits.shape)
logits = np.squeeze(logits)

#print(logits)
#pred = np.argmax(logits, axis=0)
#prob = logits[pred]
#print(pred, prob)
#logits[pred] = 0

for i in range(5):
    pred = np.argmax(logits, axis=0)
    prob = logits[pred]
    logits[pred] = 0
    print(pred, prob)


##feat = res['MobilenetV2__Conv__BatchNorm__FusedBatchNorm__0']
#feat = res['MobilenetV2__Conv__Relu6__0']
#feat = np.squeeze(feat)
#print(feat.shape)
##print(feat)
#c = 10
#featimg = np.ones((112,112))
#for i in range(112):
#  for j in range(112):
#    featimg[i,j] = feat[c,i,j] * 40
#print(featimg[0,:])
#featimg = Image.fromarray(np.uint8(featimg))
#featimg.save('feat.png')


