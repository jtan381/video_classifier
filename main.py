import cv2
from label_image import *
print(cv2.__version__)

# Label font and position
font = cv2.FONT_HERSHEY_SIMPLEX
bottomRightOfText = (800, 600)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2

videopath = './data/Wolf.mp4'


def predictVideocap(path):
    vidcap = cv2.VideoCapture(path)
    while(vidcap.isOpened()):

        # Capture frame-by-frame
        ret, frame = vidcap.read()
        if ret == True:

            results, top_label = imageClassify.predict(frame)

            cv2.putText(frame, "{}: {:.2f}%".format(labels[top_label[0]], results[top_label[0]]*100),
                        bottomRightOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)

            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break
    vidcap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # create imageClassify object
    imageClassify = create_imageClassify()
    imageClassify.load_graph()
    labels = imageClassify.load_labels()
    predictVideocap(videopath)
