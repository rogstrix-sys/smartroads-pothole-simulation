import os, sys, cv2, numpy as np

# Dynamically resolve Webots installation directory
possible_paths = [
    r'D:\project\Webots\lib\controller\python',
    r'C:\Program Files\Webots\lib\controller\python',
    os.path.join(os.environ.get('WEBOTS_HOME', ''), 'lib', 'controller', 'python')
]

for path in possible_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)
        print(f'>>> Linked Webots Controller API from: {path}')
        break

from controller import Supervisor

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Locate camera device
cam = None
for i in range(robot.getNumberOfDevices()):
    dev = robot.getDeviceByIndex(i)
    if dev.getNodeType() == 42:  # WB_NODE_CAMERA
        cam = dev
        print(f'>>> Found Camera Device: {cam.getName()} ({cam.getWidth()}x{cam.getHeight()})')
        break

if not cam:
    cam = robot.getDevice('camera')

if cam:
    cam.enable(timestep)
    # Warm up 5 physics steps
    for _ in range(5):
        robot.step(timestep)
    
    raw = cam.getImage()
    if raw is not None:
        img = np.frombuffer(raw, np.uint8).reshape((cam.getHeight(), cam.getWidth(), 4))
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        brightness = float(np.mean(bgr))
        print(f'>>> Camera Active! Average pixel brightness: {brightness:.2f}')
        if brightness < 2.0:
            print('>>> WARNING: Feed is pitch black. Ensure Webots is PLAYING and camera translation is clear of car body.')
        else:
            print('>>> SUCCESS: Camera feed is bright and receiving live road scenery!')
            cv2.imshow('Test Camera Feed', bgr)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
    else:
        print('>>> ERROR: Camera returned NULL buffer. Check if Webots is playing.')
else:
    print('>>> ERROR: No camera node found on the vehicle.')
