import face_recognition
import cv2
import time

input_video = cv2.VideoCapture('video_1_raw.mp4')

fps = int(input_video.get(cv2.CAP_PROP_FPS))
frame_count = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(fps)
print(frame_count)
print(frame_width)
print(frame_height)

codec = cv2.VideoWriter.fourcc(*'XVID')
video_writer = cv2.VideoWriter('video_1_processed.mp4', codec,fps, (frame_width, frame_height))

face_locations = []

count = 0
percentage_of_frames = 4
start = time.time()
while (True):
    ret, frame = input_video.read()
    if count % percentage_of_frames == 0:
        if not ret:
            print("Video ended!")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        face_locations = face_recognition.face_locations(rgb_frame, model='cnn')

        for top, right, bottom, left in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 225), 2)

        video_writer.write(frame)

        print('Processed ', count%percentage_of_frames, ' frames')

    count += 1

print('Result:', count)
print('Taken time: ', (time.time() - start) % 60, ' minutes')

input_video.release()
video_writer.release()
cv2.destroyAllWindows()
