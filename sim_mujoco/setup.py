from setuptools import setup
from glob import glob
import os

package_name = 'sim_mujoco'
submodules = "sim_mujoco/submodules"

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, submodules],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sorina Lupu',
    maintainer_email='lupusorina@yahoo.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mujoco = sim_mujoco.mujoco:main',
            'joy = sim_mujoco.joynode:main'
        ],
    },
)

