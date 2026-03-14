import depthai as dai
import cv2

# Create a pipeline
pipeline = dai.Pipeline()

# In depthai v3, Camera node replaces ColorCamera
# .build() is required to initialise it
cam = pipeline.create(dai.node.Camera).build()

# Request the output size we want
videoOutput = cam.requestOutput((1280, 720), type=dai.ImgFrame.Type.BGR888p)

# Create output queue directly from the output
videoQueue = videoOutput.createOutputQueue()

# Start the pipeline
print("Starting camera...")
pipeline.start()
print("Camera started! Press Q to quit.")

while pipeline.isRunning():
    frame = videoQueue.tryGet()
    if frame is not None:
        cv2.imshow("OAK-D Lite - Colour Camera", frame.getCvFrame())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("Camera stopped.")
