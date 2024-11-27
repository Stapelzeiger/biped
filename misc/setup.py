from setuptools import setup
from glob import glob
import os

package_name = 'misc'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.yml')),
    ],
    install_requires=[
        'setuptools',
        'motion_capture_tracking_interfaces',
    ],
    zip_safe=True,
    maintainer='sorina',
    maintainer_email='lupusorina@yahoo.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'misc = misc.misc:main',
            'extract_tf_base_link_base_link_mocap = misc.extract_tf_base_link_base_link_mocap:main',
        ],
    },
)
