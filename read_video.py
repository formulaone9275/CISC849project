import numpy as np
import cv2
import os
#convert video to images of experiment 1,2,3

def generate_one_image_from_video():
    video_file_path='UCF50/'
    image_file_path='data/experiment1/'
    for cate_i in os.listdir(video_file_path):
        file_count=0
        for file_i in os.listdir(video_file_path+cate_i):
            #print file_i
            cap = cv2.VideoCapture(video_file_path+cate_i+'/'+file_i)
            #7 is the parameter for the frame number
            frames_num=cap.get(7)
            count_index=0
            while(cap.isOpened()):
                ret, frame = cap.read()
                if count_index== int(frames_num/2):
                    #2000 for dev, 2000 for test, the rest for training
                    if file_count<20:
                        image_destination=image_file_path+'dev/'+cate_i
                    elif file_count>=20 and file_count<40:
                        image_destination=image_file_path+'test/'+cate_i
                    else:
                        image_destination=image_file_path+'train/'+cate_i
                    if not os.path.exists(image_destination):
                        os.makedirs(image_destination)
                    cv2.imwrite(image_destination+'/'+cate_i+'_'+str(file_count)+'.jpg', frame)
                    #print frame.shape
                    break
                count_index+=1
                #print count_index
            cap.release()
            file_count+=1

    cv2.destroyAllWindows()

def generate_multi_images_from_video(image_num,cut_num):
    video_file_path='UCF50/'
    image_file_path='data/experiment2/'
    for cate_i in os.listdir(video_file_path):
        file_count=0
        for file_i in os.listdir(video_file_path+cate_i):
            image_count=0
            #print file_i
            cap = cv2.VideoCapture(video_file_path+cate_i+'/'+file_i)
            frames_num=cap.get(7)
            #cut some frames from the beginning and ending
            frame_interval=int((frames_num-2*cut_num)/image_num)
            count_index=0
            while(cap.isOpened()):
                ret, frame = cap.read()
                if count_index== (frame_interval*image_count+cut_num):
                    #2000 for dev, 2000 for test, the rest for training
                    if file_count<20:
                        image_destination=image_file_path+'dev/'+cate_i
                    elif file_count>=20 and file_count<40:
                        image_destination=image_file_path+'test/'+cate_i
                    else:
                        image_destination=image_file_path+'train/'+cate_i
                    if not os.path.exists(image_destination):
                        os.makedirs(image_destination)
                    cv2.imwrite(image_destination+'/'+cate_i+'_'+str(file_count)+'_'+str(image_count)+'.jpg', frame)
                    image_count+=1
                    #print frame.shape
                    if image_count==image_num:
                        break
                count_index+=1
                #print count_index
            cap.release()
            file_count+=1

    cv2.destroyAllWindows()

def generate_downsample_images_from_video(down_sample_rate):
    video_file_path='UCF50/'
    image_file_path='data/experiment3/'
    for cate_i in os.listdir(video_file_path):
        file_count=0
        for file_i in os.listdir(video_file_path+cate_i):
            image_count=0
            #print file_i
            cap = cv2.VideoCapture(video_file_path+cate_i+'/'+file_i)
            frames_num=cap.get(7)
            #cut some frames from the beginning and ending
            count_index=0
            while(cap.isOpened()):
                ret, frame = cap.read()
                if count_index== (down_sample_rate*image_count):
                    #2000 for dev, 2000 for test, the rest for training
                    if file_count<20:
                        image_destination=image_file_path+'dev/'+cate_i
                    elif file_count>=20 and file_count<40:
                        image_destination=image_file_path+'test/'+cate_i
                    else:
                        image_destination=image_file_path+'train/'+cate_i
                    if not os.path.exists(image_destination):
                        os.makedirs(image_destination)
                    cv2.imwrite(image_destination+'/'+cate_i+'_'+str(file_count)+'_'+str(image_count)+'.jpg', frame)
                    image_count+=1
                    #print frame.shape
                    if image_count==int(frames_num/down_sample_rate):
                        break
                count_index+=1
                #print count_index
            cap.release()
            file_count+=1

    cv2.destroyAllWindows()

if __name__=='__main__':
    #generate the data for all the three experiments
    generate_one_image_from_video()
    generate_multi_images_from_video(5,10)
    generate_downsample_images_from_video(6)