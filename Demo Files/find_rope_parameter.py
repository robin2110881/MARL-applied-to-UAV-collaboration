import pybullet as p
import pybullet_data
import numpy as np
import time

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -9.81)
p.setTimeStep(1.0 / 240.0)

plane = p.loadURDF("plane.urdf")

# -----------------------------------------------------------------------------
# Boxes
# -----------------------------------------------------------------------------

box_half = [0.2, 0.2, 0.2]

box_collision = p.createCollisionShape(
    p.GEOM_BOX,
    halfExtents=box_half
)

box_visual = p.createVisualShape(
    p.GEOM_BOX,
    halfExtents=box_half
)

box1 = p.createMultiBody(
    baseMass=0.05,
    baseCollisionShapeIndex=box_collision,
    baseVisualShapeIndex=box_visual,
    basePosition=[0, 0, 0.2]
)

box2 = p.createMultiBody(
    baseMass=0.05,
    baseCollisionShapeIndex=box_collision,
    baseVisualShapeIndex=box_visual,
    basePosition=[1.5, 0, 0.2]
)

# -----------------------------------------------------------------------------
# Rope model
# -----------------------------------------------------------------------------

REST_LENGTH = 2.0
STIFFNESS = 100.0
DAMPING = 1.5
MAX_TENSION = 20.0  # optional safety clamp

rope_line = -1

# -----------------------------------------------------------------------------
# Give box2 an initial velocity
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

while p.isConnected():

    pos1, _ = p.getBasePositionAndOrientation(box1)
    pos2, _ = p.getBasePositionAndOrientation(box2)

    vel1, _ = p.getBaseVelocity(box1)
    vel2, _ = p.getBaseVelocity(box2)

    pos1 = np.array(pos1)
    pos2 = np.array(pos2)

    vel1 = np.array(vel1)
    vel2 = np.array(vel2)

    delta = pos2 - pos1
    distance = np.linalg.norm(delta)

    if distance > 1e-8:

        direction = delta / distance

        extension = distance - REST_LENGTH

        if extension > 0:

            # Relative speed along rope axis
            rel_speed = np.dot(
                vel2 - vel1,
                direction
            )

            tension = (
                STIFFNESS * extension +
                DAMPING * rel_speed
            )

            tension = max(0.0, tension)
            tension = min(MAX_TENSION, tension)

            force = tension * direction

            p.applyExternalForce(
                objectUniqueId=box1,
                linkIndex=-1,
                forceObj=force.tolist(),
                posObj=pos1.tolist(),
                flags=p.WORLD_FRAME
            )

            p.applyExternalForce(
                objectUniqueId=box2,
                linkIndex=-1,
                forceObj=(-force).tolist(),
                posObj=pos2.tolist(),
                flags=p.WORLD_FRAME
            )

    # Draw rope
    rope_line = p.addUserDebugLine(
        pos1,
        pos2,
        lineColorRGB=[1, 0, 0],
        lineWidth=3,
        replaceItemUniqueId=rope_line
    )

    p.stepSimulation()
    time.sleep(1.0 / 240.0)