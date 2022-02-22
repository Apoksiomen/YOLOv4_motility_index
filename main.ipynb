from yolo import YoloDetection  #ovo je za napraviti model
from sort import *              #ovo je SORT
import matplotlib.pyplot as plt #za graf
import numpy as np
import cv2                      #opencv
import argparse
from math import pi
import time

# Create tracker object
#tracker = EuclideanDistTracker()
mot_tracker=Sort()

#graf
x = np.linspace(1, 100, 100)
y = np.linspace(1, 2000, 100)
plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x, y, 'r-') # Returns a tuple of line objects, thus the comma
 
# setting labels
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Updating plot...")

CONFIG_FILE = None
model = None

def load_config(config_path):
    global CONFIG_FILE
    CONFIG_FILE = eval(open(config_path).read())
def load_model():
    global model
    model = YoloDetection(CONFIG_FILE["model-parameters"]["model-weights"],
                    CONFIG_FILE["model-parameters"]["model-config"],
                    CONFIG_FILE["model-parameters"]["model-names"],
                    CONFIG_FILE["shape"][0],
                    CONFIG_FILE["shape"][1])

def start_detection(media_path):
    #ovaj threshold je pomak iznad kojeg se zbraja
    threshold=8
    dist_travelled=1
    brojac=1
    detection_prev=pd.DataFrame()
    xdata, ydata=[],[]    
    prev=0#za mjerenje fps
    cv2.namedWindow("Video",cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(media_path)
    ret = True

    while ret:
        
        ret , frame = cap.read()
        if(ret):
            #ovo vraća x,y,x1,y1,confidence
            detections = np.array(model.process_frame(frame))
            boxes_ids = mot_tracker.update(detections)

            if brojac>1:
                skupna=pd.merge(pd.DataFrame(boxes_ids), detection_prev, left_on=4, right_on=4)
                #skupna=skupna.apply(pd.to_numeric)
                #skupna= skupna.abs()
                for j in range(len(skupna)):
                    coords=boxes_ids[j]
                    #x1,y1,x2,y2,name_idx=int(coords[0]),int(coords[1]),int(coords[2]),int(coords[3]),int(coords[4])
                    x1,y1,x2,y2,name_idx=int(skupna.iloc[j,0]),int(skupna.iloc[j,1]),\
                    int(skupna.iloc[j,2]),int(skupna.iloc[j,3]),int(skupna.iloc[j,4])
                    #sredine
                    mid_x=int((x1+x2)/2)
                    mid_y=int((y1+y2)/2)
                    mid_x2=int(skupna.iloc[j,5]+skupna.iloc[j,7])/2
                    mid_y2=int(skupna.iloc[j,6]+skupna.iloc[j,8])/2
                    
                    pomak=math.sqrt(abs((mid_x-mid_x2)**2 + (mid_y-mid_y2)**2))
                    if pomak>threshold:
                        dist_travelled=int(dist_travelled+pomak)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),thickness=2,color=(255,0,0))
                    cv2.circle(frame,(mid_x,mid_y),10, color=(0,0,255), thickness=2)
                    cv2.putText(frame,str(name_idx),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    cv2.putText(frame,str(int(pomak)),(mid_x,mid_y),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    cv2.putText(frame,str(brojac),(10,35),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            cv2.putText(frame,str(dist_travelled),(10,155),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow("Video",frame)  
            detection_prev=pd.DataFrame(boxes_ids)
              
            brojac=brojac+1
            key = cv2.waitKey(30)
            ## quit detection when esc key pressed
            if(key==27):
                break

            ## paused detection when space key pressed
            if(key==32):
                cv2.waitKey(-1)    

            #graf
            if(brojac/150).is_integer():
                
                plt. clf() 
                x = np.linspace(1, int(brojac*2), 100)
                y = np.linspace(1, int(dist_travelled*2), 100)
                plt.ion()
                ax = fig.add_subplot(111)
                line1, = ax.plot(x, y, 'r-') # Returns a tuple of line objects, thus the comma                
                xdata.append(brojac)
                ydata.append(dist_travelled)
                line1.set_data(xdata, ydata)
                time_elapsed = int((time.time() - prev))
                #title_s=str(brojac)
                plt.xlabel("Framova")
                plt.ylabel("Piksela ukupno")
                plt.title("Framova: "+str(brojac)+"   "+"FPS: "+str(round(150/time_elapsed,2)))
                prev = time.time()
                fig.canvas.draw()
                fig.canvas.flush_events()

    cv2.destroyAllWindows()
"""
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Provide arguements")
    parser.add_argument("--config","-c")
    parser.add_argument("--debug","-d")
    parser.add_argument("--video","-v")
    args = parser.parse_args()
    config_path = args.config
    load_config(config_path)
    load_model()
    start_detection(args.video)
"""
load_config("config.json")
load_model()
start_detection("movies\hiv00175.mp4")
