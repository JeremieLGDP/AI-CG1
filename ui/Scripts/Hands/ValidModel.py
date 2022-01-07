import matplotlib.pyplot as plt
import torch
import joblib
import torch.nn as nn
import numpy as np
import cv2
import albumentations
import torch.nn.functional as F
import time
import CustomModelCNN

test_img=str(input("Enter the path of test image :"))
#print(test_img)


aug = albumentations.Compose([
                albumentations.Resize(224, 224, always_apply=True),
])

# load label binarizer
lb = joblib.load('output/lb.pkl')


model = CustomModelCNN.CustomCNN()
model.load_state_dict(torch.load('output/best.pth'))
print(model)
print('Model loaded')


image = cv2.imread(test_img)
cv2.imshow('image', image)
cv2.waitKey(0)
image_copy = image.copy()
 
image = aug(image=np.array(image))['image']
image = np.transpose(image, (2, 0, 1)).astype(np.float32)
image = torch.tensor(image, dtype=torch.float)
image = image.unsqueeze(0)
print(image.shape)


start = time.time()
outputs = model(image)
_, preds = torch.max(outputs.data, 1)
print('PREDS', preds)
print(f"Predicted output: {lb.classes_[preds]}")
end = time.time()
print(f"{(end-start):.3f} seconds")
 
cv2.putText(image_copy, lb.classes_[preds], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

plt.imshow(image_copy, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
#cv2.imshow('image', image_copy)
#cv2.waitKey(0)