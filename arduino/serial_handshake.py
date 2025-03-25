import serial 
import time
import numpy as np

ar = np.random.randint(0,250,500)

s = serial.Serial('/dev/ttyACM0',921600,timeout = None)

print('start')
a = 10

for i in range(500):
    # print(a)
    # s.write(bytes(str(a+100)+'\n','utf-8'))
    t = time.time()
    a = ar[i]
    s.write(bytes(str(a+100),'utf-8'))
    # s.write(bytes(str(a+0),'utf-8'))
    time.sleep(0.01)
    aa = s.read(3) 
    # y = int.from_bytes(s.read(3),'big')
    y = int(aa)-100
    
    # print(aa)
    print(time.time()-t)
    # s.reset_output_buffer()
    # if a>202:
    #     a=0

    if y == a:
        print(1)
    else:
        print(0)


s.close()    

