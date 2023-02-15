from setuptools import setup
from glob import glob
import os

package_name = 'logging_biped'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/logging.launch.yml')),
        (os.path.join('share', package_name), glob('config/topics.yml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sorina',
    maintainer_email='lupusorina@yahoo.com',
    description='Bag Logging',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'logging_biped = logging_biped.logging_biped:main',
        ],
    },
)
