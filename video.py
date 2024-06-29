from object_detection import ObjectDetection

def main():

    video_path = 'test4.mp4'
    video = ObjectDetection()
    video.vid_display(video_path)

if __name__=="__main__":
  main()
