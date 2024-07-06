import os
import cv2

phases = ['Train', 'Test', 'Validation']

for phase in phases:
    # path = os.path.join('/path/to/video/files/', phase)
    path = os.path.join('Dataset', phase) # path = Dataset\Train
    subjects = os.listdir(path)

    for subject in subjects:
        print(phase, subject, flush=True)
        static_path = f"Dataset\\{phase}_Frames\\{subject}"
        if(os.path.exists(static_path)):
            print(f"{static_path} subject skipped...")
            continue
            
        videos = os.listdir(os.path.join(path, subject))
        for video in videos:
            print(f"Video in {os.path.join(path, subject, video)}: {os.listdir(os.path.join(path, subject, video))}")
            videoPath = os.path.join(path, subject, video, os.listdir(os.path.join(path, subject, video))[0])
            # print(f"Video path: {videoPath}") # OUTPUT: Video path: Dataset\Train\110001\1100011002\1100011002.avi
            currentVideoName = videoPath.split('\\')[-1]
            videoPathFrames = '\\'.join(videoPath.split('.')[0].split('\\')[:-1]).replace(phase, phase + '_Frames') # Location to save each video's frame
            # print(f"videoPathFrames: {videoPathFrames}") # OUTPUT: videoPathFrames: Dataset\Train_Frames\110001\1100011002
            os.makedirs(videoPathFrames, exist_ok=True) # Directory: Dataset\Train_Frames\110001\1100011002 created

            capture = cv2.VideoCapture(videoPath) # Read each video
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            count = 0
            i = 0
            retaining = True
            while count < frame_count and retaining:
                retaining, frame = capture.read() # Read each frame of a video
                
                if frame is None:
                    continue
                cv2.imwrite(filename=os.path.join(videoPathFrames, '{}.jpg'.format(str(i))), img=frame)
                print(f"Video {currentVideoName} frame '{str(i)}'.jpg saved to: {videoPathFrames}")
                i += 1
                count += 1
            capture.release()
