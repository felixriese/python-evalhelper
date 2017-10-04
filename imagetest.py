#imagetest.py

import numpy as np
from PIL import Image
np.set_printoptions(threshold=np.nan)

testbox = np.array(Image.open("crosses/work_type_crossed/P_24.2863___A3yz2pa0018.jpg__23747.png").convert('L').getdata())
testbox2 = np.array(Image.open("crosses/work_type_crossed/P_24.5765___A3yz2pa0018.jpg__23805.png").getdata())

print(testbox)
#print(testbox2)