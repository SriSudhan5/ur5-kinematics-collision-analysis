from setuptools import find_packages, setup

package_name = 'ur5_traj_tools'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robotics',
    maintainer_email='robotics@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'traj_playback = ur5_traj_tools.traj_playback:main',
        'traj_check = ur5_traj_tools.traj_check:main',
        'angle_sender = ur5_traj_tools.angle_sender:main',
        'collision_map_joint = ur5_traj_tools.collision_map_joint:main',
        'pose_sender = ur5_traj_tools.pose_sender:main',

    ],
},

)
