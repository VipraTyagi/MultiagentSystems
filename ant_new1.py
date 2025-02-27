import blueprints as blue
import numpy as np

world = blue.World()

# Create bodies
head = blue.Body(name='head')
thorax = blue.Body(name='thorax')
abdomen = blue.Body(name='abdomen')

left_front_leg = blue.Body(name='left_front_leg')
right_front_leg = blue.Body(name='right_front_leg')
left_middle_leg = blue.Body(name='left_middle_leg')
right_middle_leg = blue.Body(name='right_middle_leg')
left_rear_leg = blue.Body(name='left_rear_leg')
right_rear_leg = blue.Body(name='right_rear_leg')

# Create geoms
head_geom = blue.geoms.Sphere( radius=0.1)
thorax_geom = blue.geoms.Capsule(radius=0.1, length=0.5)
abdomen_geom = blue.geoms.Capsule(radius=0.1, length=0.5)

leg_geom = blue.geoms.Capsule(radius=0.05, length=0.5)

# Attach geoms to bodies
head.attach(head_geom)
thorax.attach(thorax_geom)
abdomen.attach(abdomen_geom)

for leg in [left_front_leg, right_front_leg, left_middle_leg, right_middle_leg, left_rear_leg, right_rear_leg]:
    leg.attach(leg_geom.copy())

# Create joints
neck_joint = blue.joints.Joint(
    pos=[0, 0, 0],
    axis=[0, 1, 0],
    range=[-np.pi/2, np.pi/2],
    ref=0,
    frictionloss=0.0,
    name="neck_joint"
)

abdominal_joint = blue.joints.Joint(
    pos=[0, 0, -0.25],
    axis=[0, 1, 0],
    range=[-np.pi/2, np.pi/2],
    ref=0,
    frictionloss=0.0,
    name="abdominal_joint"
)

# Attach joints to appropriate bodies
head.attach(neck_joint)
thorax.attach(abdominal_joint)

# Create leg joints
leg_joints = [
    blue.joints.Joint(pos=[0, 0, 0], axis=[0, 1, 0], range=[-np.pi/2, np.pi/2], ref=0, frictionloss=0.0),
    blue.joints.Joint(pos=[0, 0, 0], axis=[0, 1, 0], range=[-np.pi/2, np.pi/2], ref=0, frictionloss=0.0),
    blue.joints.Joint(pos=[0, 0, 0], axis=[0, 1, 0], range=[-np.pi/2, np.pi/2], ref=0, frictionloss=0.0),
    blue.joints.Joint(pos=[0, 0, 0], axis=[0, 1, 0], range=[-np.pi/2, np.pi/2], ref=0, frictionloss=0.0),
    blue.joints.Joint(pos=[0, 0, 0], axis=[0, 1, 0], range=[-np.pi/2, np.pi/2], ref=0, frictionloss=0.0),
    blue.joints.Joint(pos=[0, 0, 0], axis=[0, 1, 0], range=[-np.pi/2, np.pi/2], ref=0, frictionloss=0.0)
]

# Attach legs to thorax
thorax.attach(left_front_leg)
thorax.attach(right_front_leg)
thorax.attach(left_middle_leg)
thorax.attach(right_middle_leg)
thorax.attach(left_rear_leg)
thorax.attach(right_rear_leg)

# Attach leg joints to legs
left_front_leg.attach(leg_joints[0])
right_front_leg.attach(leg_joints[1])
left_middle_leg.attach(leg_joints[2])
right_middle_leg.attach(leg_joints[3])
left_rear_leg.attach(leg_joints[4])
right_rear_leg.attach(leg_joints[5])

# Create actuators
neck_actuator = blue.actuators.General(
    joint=neck_joint,
    ctrllimited=True,
    ctrlrange=[-np.pi/2, np.pi/2],
    
    
)
abdominal_actuator = blue.actuators.General(
    joint=abdominal_joint,
    ctrllimited=True,
    ctrlrange=[-np.pi/2, np.pi/2],
    
    
)

head.attach(neck_actuator)
thorax.attach(abdominal_actuator)

leg_actuators = [
    blue.actuators.General(joint=joint, ctrl_range=(-np.pi/2, np.pi/2)) for joint in leg_joints
]

left_front_leg.attach(leg_actuators[0])
right_front_leg.attach(leg_actuators[1])
left_middle_leg.attach(leg_actuators[2])
right_middle_leg.attach(leg_actuators[3])
left_rear_leg.attach(leg_actuators[4])
right_rear_leg.attach(leg_actuators[5])

# Attach bodies to world
world.attach(head)
world.attach(thorax)
world.attach(abdomen)
world.attach(left_front_leg)
world.attach(right_front_leg)
world.attach(left_middle_leg)
world.attach(right_middle_leg)
world.attach(left_rear_leg)
world.attach(right_rear_leg)



# Visualize the world
world.view()
