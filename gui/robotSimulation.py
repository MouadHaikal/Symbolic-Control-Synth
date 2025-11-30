import pybullet as p
import pybullet_data
import time
import math
import random

def celebrateAtGoal(robotId, goal_pos, dance_duration=4, num_balloons=15):
    """
    Make the robot dance while turning on itself with balloons rising.
    """
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
    """
    Simulate a robot following a path with obstacles and goal. Dance at goal.
    """

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

    boxSize = [0.2, 0.2, 0.2]
    col_box = p.createCollisionShape(p.GEOM_BOX, halfExtents=boxSize)
    vis_box = p.createVisualShape(p.GEOM_BOX, halfExtents=boxSize, rgbaColor=[1, 0, 0, 1])

    for obs in obstacles:
        (xmin, ymin), (xmax, ymax) = obs
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        cz = boxSize[2]
        p.createMultiBody(0, col_box, vis_box, [cx, cy, cz])

    (gxMin, gyMin), (gxMax, gyMax) = goal
    goal_pos = [(gxMin + gxMax)/2, (gyMin + gyMax)/2, 0.15]

    goalR = 0.25
    goalCol = p.createCollisionShape(p.GEOM_SPHERE, radius=goalR)
    goalVis = p.createVisualShape(p.GEOM_SPHERE, radius=goalR, rgbaColor=[0, 1, 0, 1])
    p.createMultiBody(0, goalCol, goalVis, goal_pos)

    currentPos = startPos
    steps = 500 
    for target in path:
        target_pos = [target[0], target[1], 0.12]

        for i in range(steps):
            t = i / (steps - 1)
            interp = [
                currentPos[j] + (target_pos[j] - currentPos[j]) * t
                for j in range(3)
            ]
            p.resetBasePositionAndOrientation(robotId, interp, startOrientation)
            p.stepSimulation()
            time.sleep(1./480.)

        currentPos = target_pos

    print("Goal reached: Robot is spinning and dancing with balloons!")
    celebrateAtGoal(robotId, goal_pos, dance_duration=5, num_balloons=20)

    while True:
        p.stepSimulation()
        time.sleep(1./240.)


# ---------------------- Example usage ----------------------
start = (0, 0)
obstacles = [
    [(1, 1), (2, 2)],
    [(3, 3), (4, 4)]
]
goal = [(5, 5), (5, 5)]
path = [
    (0, 2),
    (2, 2.5),
    (4, 2.5),
    (5, 5)
]


simulateRobot(start, obstacles, goal, path)
