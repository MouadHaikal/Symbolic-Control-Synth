import pybullet as p
import pybullet_data
import time
import math
import random

def celebrateAtGoal(robotId, goal_pos, dance_duration=4, num_balloons=15):
    start_time = time.time()
    balloons = []
    colors = [[1,0,0,1],[0,1,0,1],[0,0,1,1],[1,1,0,1],[1,0,1,1],[0,1,1,1]]
    for _ in range(num_balloons):
        x = goal_pos[0] + random.uniform(-0.3,0.3)
        y = goal_pos[1] + random.uniform(-0.3,0.3)
        z = goal_pos[2] + 0.3
        color = random.choice(colors)
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=color)
        balloon = p.createMultiBody(0.01, col, vis, [x, y, z])
        balloons.append(balloon)

    while time.time() - start_time < dance_duration:
        t = time.time() * 2  
        roll = 0.05 * math.sin(t*3)   
        pitch = 0.1 * math.sin(t*2)   
        yaw = (t * 15) % (2*math.pi)         
        orientation = p.getQuaternionFromEuler([roll, pitch, yaw])
        pos, _ = p.getBasePositionAndOrientation(robotId)
        p.resetBasePositionAndOrientation(robotId, pos, orientation)

        for b in balloons:
            pos_b, orn_b = p.getBasePositionAndOrientation(b)
            new_pos = [pos_b[0], pos_b[1], pos_b[2] + 0.002]
            p.resetBasePositionAndOrientation(b, new_pos, orn_b)

        p.stepSimulation()
        time.sleep(1./240.)

    for b in balloons:
        p.removeBody(b)

def simulateRobot(startState, obstacles, goal, path, robot_urdf="r2d2.urdf"):
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    startPos = [startState[0], startState[1], 0.12]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    robotId = p.loadURDF(robot_urdf, startPos, startOrientation)

    for obs in obstacles:
        (xmin, ymin), (xmax, ymax) = obs
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        cz = 0.2  
        half_extents = [(xmax - xmin)/2, (ymax - ymin)/2, cz]
        col_box = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis_box = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[1, 0, 0, 1])
        p.createMultiBody(0, col_box, vis_box, [cx, cy, cz])

    gxMin, gyMin = goal[0]
    gxMax, gyMax = goal[1]
    goal_cx = (gxMin + gxMax) / 2
    goal_cy = (gyMin + gyMax) / 2
    goal_cz = 0.2
    goal_half_extents = [(gxMax - gxMin)/2, (gyMax - gyMin)/2, goal_cz]
    goal_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=goal_half_extents)
    goal_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=goal_half_extents, rgbaColor=[0, 1, 0, 1])
    p.createMultiBody(0, goal_col, goal_vis, [goal_cx, goal_cy, goal_cz])
    goal_pos = [goal_cx, goal_cy, goal_cz]

    currentPos = startPos
    steps = 500 
    for target in path:
        target_pos = [target[0], target[1], 0.12]
        for i in range(steps):
            t = i / (steps - 1)
            interp = [currentPos[j] + (target_pos[j] - currentPos[j]) * t for j in range(3)]
            p.resetBasePositionAndOrientation(robotId, interp, startOrientation)
            p.stepSimulation()
            time.sleep(1./480.)
        currentPos = target_pos

    print("Goal reached: Robot is spinning and dancing with balloons!")
    celebrateAtGoal(robotId, goal_pos, dance_duration=5, num_balloons=20)

    while True:
        p.stepSimulation()
        time.sleep(1./240.)

