import serial 
import time
s = serial.Serial('/dev/ttyACM0',921600,timeout = 0)
print('start')
a = 10

for i in range(200):
    # print(a)
    # s.write(bytes(str(a+100)+'\n','utf-8'))
    t = time.time()
    s.write(bytes(str(a+100),'utf-8'))
    time.sleep(0.0285)
    print(time.time()-t)
    a = a+10
    # s.reset_output_buffer()
    if a>202:
        a=0


s.close()    

