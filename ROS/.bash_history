ls
cd ..
cd catkin_ws/
ls
catkin_make
ls
clear
cd /opt/
cd ..
ls
cd /opt/ros/indigo/lib/ps3joy/
ls
sudo vi ps3joy.py 
cd /opt/ros/indigo/share/fetch_navigation/
ls
cd launch/
ls
sudo vi freight_nav.launch 
roslaunch fetch_navigation freight_nav.launch 
roslaunch robot_local_control robot_local_control.launch 
rosrun patrolling patrol.py
roslaunch rostful rostful_server.launch 
ls
cd catkin_ws/
ls
cd src/
ls
sudo apt-get install ros-indigo-rosbridge-suite
source /opt/ros/indigo/setup.bash
roslaunch rosbridge_server rosbridge_websocket.launch
ls
git clone https://github.com/RobotWebTools/roslibjs.git
ls
cd ..
catkin_make
cd src/
ls
cd roslibpy/
ls
cd doc
cd ..
cd roslibpy/
ls
cd docker/
ls
cd ..
ls
cd src/
ls
cd roslibpy
ls
cd ../..
ls
cd ..
cd roslibjs/
ls
cd src/
ls
cd ..
ls
cd build/
ls
cd ..
ls
cd examples/
ls
cd ..
cd test/
ls
cd tcp/
ls
cd ..
ls
cd workersocket/
ls
cd ..
ls
cd examples/
ls
cd ..
ls
cd ..
cd roslibpy
ls
cd tests/
ls
cd ..
cd ../roslibjs/
l
ls
cd examples/
ls
cd ../..
cd rosbridge_suite/
ls
cd /opt/ros/indigo/
ls
cd lib/
ls
cd rosbridge_server/
ls
cd ..
cd share/
ls
cd rosbridge_suite/
ls
cd ..
cd rosbridge_server
ls
cd launch/
ls
cd ..
cd rosbridge_suite/
ls
cd ~/
ls
cd catkin_ws/
cd src/
ls
cd rosbridge_suite/
ls
cd rosbridge_library/
ls
cd srv
ls
cd ..
ls
cd ..
ls
cd rosbridge_s
cd rosbridge_server
ls
cd scripts/
ls
cd ../src/
ls
cd backports/
ls
cd ..
cd rosbridge_server/
ls
cd ../..
ls
cd ..
ls
cd ..
ls
cd roslibjs/
ls
cd examples/
ls
touch test.html
sudo vi test.html
sudo apt-get install ros-indigo-ros-base
sudo apt-get install ros-indigo-rosbridge-server
sudo apt-get install ros-indigo-common-tutorials
ls
sudo apt-get install ros-indigo-rospy-tutorials
roscore
exit
cd /home/fetch/catkin_ws/src/rosbridge_suite/
ls
cd rosapi
ls
vi README.md
cd ..
ls 
cd /opt/ros/indigo/share/
ls
cd rosbridge_server/
ls
cd ..
cd ~roscore
roscore
cd ~/catkin_ws/
ls
cd src/
ls
cd rosbridge_suite/
ls
cd rosbridge_
cd rosbridge_server/
ls
cd launch/
cd ..
cd launch/
ls
roslaunch rosbridge_suite rosbridge_websocket.launch
roslaunch rosbridge_server rosbridge_websocket.launch
roscore
rostopic echo /cmd_vel
rosrun rospy_tutorials add_two_ints_server
killall -9 roscore
sudo killall -9 roscore
killall -9 rosmaster
sudo killall -9 rosmaster
roscore
pip install roslibpy
pip install -r requirements-dev.txt
cd catkin_ws/
ls
cd src/
ls
cd roslibpy/
ls
pip install -r requirements-dev.txt
invoke
pyinvoke
ls
python setup.py
python setup.py build
python setup.py install
ls
vi tasks.py 
vi setup.py 
rostopic pub /listener std_msgs/String "Hello World"
cd /opt/ros/indigo/share/
cd rosbridge_server/
ls
cd launch/
ls
roslaunch rosbridge_server rosbridge_websocket.launch 
roslaunch rostful rostful_server.launch 
ls
vi rosbridge_websocket.launch 
clear
roslaunch rostful rostful_server.launch 
roslaunch rosbridge_server rosbridge_websocket.launch 
cd /opt/ros/indigo/share/
ls
cd rosbridge_server/
ls
cd
cd ~/catkin_ws/
ls
cd src/
ls
cd rosbridge_suite/
ls
cd rosbridge_server/
ls
roslaunch fetch_navigation freight_nav.launch 
roslaunch rosbridge_server rosbridge_websocket.launch 
rostopic echo /odom
eixt
cd ~/catkin_ws/
ls
cd src/
ls
cd check_odometry/
ls
cd launch/
ls
sudo vi check_odom.launch 
roslaunch check_odom check_odom.launch
ls
roslaunch check_odometry check_odom.launch 
rostopic type /cmd_vel
rostopic type /amcl_pose
rosmsg show geometry_msgs/Twist
rostopic echo /cmd_vel
roslaunch check_odometry check_odom.launch 
l
cd ../src/
ls
cd ../launch/
ls
sudo vi check_odom.launch 
cd ../src/
sudo vi check_odom.py 
roslaunch check_odometry check_odom.launch 
cd ..
ls
sudo vi package.xml 
exit
roslaunch fetch_navigation freight_nav.launch 
exit
roslaunch check_odometry check_odom.launch 
cd ~/catkin_ws/src/
ls
cd check_odometry/
ls
cd launch/
cd ..
cd src/
ls
vi check.py 
cd /opt/ros/indigo/share/nav_msgs/
ls
cd msg/
ls
vi Odometry.msg 
rostopic echo /odom
roslaunch fetch_navigation freight_nav.launch 
rosrun sound_play say.py "seung won ah"
rosrun sound_play say.py "seung won ah"\
rosrun sound_play say.py "seung won ah"
ls
cd /opt/ros/indigo/lib/
ls
cd sound_play/
ls
vi shutup.py 
sudo vi say.py
sudo vi soundplay_node.py
ls
rostopic echo /odom
roslaunch rosbridge_server rosbridge_websocket.launch 
roslaunch check_odometry check_odom.launch 
roslaunch fetch_navigation freight_nav.launch 
cd ~/catkin_ws/
ls
cd src/
cd check_odometry/
ls
cd src/
ls
cd ..
cd launch/
ls
sudo vi check_odom.launch 
roslaunch check_odometry check_odom.launch 
sudo vi check_odom.launch 
roslaunch check_odometry check_odom.launch 
sudo vi check_odom.launch 
cd ..
catkin_create_pkg check_odometry
catkin_make
cd ..
catkin_make
roslaunch check_odometry check_odom.launch 
cd src/check_odometry/
ls
sudo vi CMakeLists.txt 
roslaunch check_odometry check_odom.launch 
sudo vi CMakeLists.txt 
cd ~/catkin_ws/
source devel/setup.bash
roslaunch check_odometry check_odom.launch 
cd src/check_odometry/
ls
cd src/
ls
chmod +x check_odom.py 
ls -la
mv check_odom.py check.py
cd ../launch/
ls
sudo vi check_odom.launch 
roslaunch check_odometry check_odom.launch 
cd sr
cd ../src/
ls
sudo vi check.py 
roslaunch check_odometry check_odom.launch 
vi check.py 
history
rostopic info /odom
rosmsg show nav_msgs/Odometryy
rosmsg show nav_msgs/Odometry
rostopic echo /Odom
rostopic echo /odom
cd ls
cd 
~/catkin_ws/src/check_odometry/src/
cd ~/catkin_ws/src/check_odometry/src/
ls
vi check.py 
cd /opt/ros/indigo/share/
cd sound_play/
ls
cd action/
ls
cd ..
cd cmake/
ls
rosrun sound_play soundplay_node.py
ls
cd catkin_ws/
ls
cd src/
ls
cd check_odometry/
ls
cd src/
ls
cd ..
ls
cd launch/
ls
cd ~/catkin_ws/src/
ls
cd check_odometry/
ls
cd launch/
ls
vi launch
ls
mv launch check_odom.launch
roslaunch check_odometry check_odom.launch 
cd ..
ls
cd src
ls
cd ..
cleear
vi check_odom.py
cd src/
ls
mv script check_odom.py
roslaunch check_odometry check_odom.launch 
cd ..
ls
cd launch/
ls
sudo vi check_odom.launch
roslaunch rosbridge_server rosbridge_websocket.launch 
cd ~/catkin_ws/src/
catkin_create_pkg
cd ~/catkin_ws/src
catkin_create_pkg check_odometry
cd check_odometry
mkdir src launch
cd src
vi script
cd ..
cd launch/
ls
vi launch
roslaunch check_odometry check_odom.launch 
cd ~/catkin_ws/
ls
cd src/
ls
cd check_odometry/
l
cd src/
ls
sudo vi check_odom.py 
rostopic echo /odom -n1
rostopic list
roslaunch check_odometry check_odom.launch 
roslaunch rosbridge_server rosbridge_websocket.launch 
rosrun teleop_twist_keyboard teleop_twist_keyboard.py 
pip uninstall dlib
dlib --version
dlib -V
cd C:\Users\hsw08\Documents\카카오톡 받은 파일\dlib-19.8.1-cp36-cp36m-win_amd64
cd C:\Users\hsw08\Documents\카카오톡 받은 파일
cd C:\Users\hsw08\Documents
cd ~/C:\Users\hsw08\Documents\카카오톡 받은 파일
exit
roslaunch check_odometry check_odom.launch 
rosrun teleop_twist_keyboard teleop_twist_keyboard.py 
roslaunch fetch_navigation freight_nav.launch 
exit
cd ca
cd catkin_ws/src/
ls
cd sh
cd check_odometry/
ls
cd src/
ls
sudo vi check.py
roslaunch check_odometry check_odom.launch
sudo vi check.py
roslaunch check_odometry check_odom.launch
sudo vi check.py
roslaunch check_odometry check_odom.launch
sudo vi check.py
roslaunch check_odometry check_odom.launch
sudo vi check.py
roslaunch check_odometry check_odom.launch
cd catkin_ws/src/
ls
cd check_odometry/
cd launch/
ls
cd ..
cd src/
ls
yes
ls
yes
ls
roslaunch check_odometry check_odom.launch
sudo vi check.py 
roslaunch check_odometry check_odom.launch
sudo vi check.py 
roslaunch check_odometry check_odom.launch
sudo vi check.py 
roslaunch check_odometry check_odom.launch
sudo vi check.py 
roslaunch check_odometry check_odom.launch
rosrun patrolling patrol.py
cd /opt/ros/indigo/
ls
cd share/
ls
cd nav_msgs/
ls
cd msg/
ls
cd ../..
cd sound_play/
ls
cd msg/
ls
cd Desktop/
ls
mv ~/Desktop/KMS.wav /opt/ros/indigo/share/sound_play/sounds
sudo mv ~/Desktop/KMS.wav /opt/ros/indigo/share/sound_play/sounds
rosrun teleop_twist_keyboard teleop_twist_keyboard.py 
cd ~/catkin_ws/src/check_odometry/
ls
cd src/
ls
rm say.pyc
roslaunch check_odometry check_odom.launch
vi check.py 
ls
rm say.py say.pyc
l
roslaunch check_odometry check_odom.launch 
roslaunch check_odometry  check_odom.launch 
vi check.py 
roslaunch check_odometry  check_odom.launch 
vi check.py 
roslaunch check_odometry check_odom.launch 
vi check.py
cd ~/catkin_ws/src/check_odometry/src/
vi check.py
roslaunch check_odometry check_odom.launch 
roslaunch check_odometry check_odom.launch vi check.py
vi check.py
roslaunch check_odometry check_odom.launch vi check.py
roslaunch check_odometry check_odom.launch
l
cd ..
cd catkin_ws/src/
ls
cd check_odometry/
ls
cd src/
ls
sudo vi check.py 
ls
rosrun check_odometry check.py
rosrun check_odometry check_odom.launch
roslaunch check_odometry check_odom.launch
cp /opt/ros/indigo/lib/sound_play/say.py ~/catkin_ws/src/check_odometry/src/
ls
sudo vi say.py
rosrun check_odometry say.py 'ya'
cd ..
cd launch/
sudo vi check_odom.launch 
roslaunch check_odometry check_odom.launch
sudo vi check_odom.launch 
roslaunch check_odometry check_odom.launch
cd ../src/
sudo vi say.py 
rosrun check_odometry say.py
cd catkin_ws/src/check_odometry/
ls
cd src/
ls
sudo vi say.py 
rosrun check_odometry say.py 'ya'
sudo vi say.py 
rosrun check_odometry say.py 'ya'
rosrun check_odometry say.py
cd catkin_ws/src/check_odometry/src/
ls
sudo vi say.py 
exit
cd /opt/ros/indigo/share/sound_play/
ls
cd sounds/
ls
cd ..
ls
roslaunch sound_play soundplay_node.launch
ls
cd sounds/
ls
rosrun sound_play play.py KMS.wav
exit
rosrun sound_play play.py --help
rosrun sound_play play.py say-beep.wav
cd /opt/ros/indigo/lib/sound_play/
ls
rosrun sound_play soundplay_node.py
rosrun sound_play play.py say-beep.wavls
rosrun sound_play play.py
ls
sudo vi say.py
exit
rosrun check_odometry say.py 'ya'
exit
cd /opt/ros/indigo/share/
cd sound_play/
l
cd ..
cd ../lib/
l
cd sound_play/
l
vi say.py
cd ~/catkin_ws/
ls
cd /opt/ros/indigo/lib/
l
cd sound_play/
ls
vi say.py 
exit
rosrun soundplay say.py "annyeong"
rosrun sound_play say.py "hi"
cd /opt/ros/indigo/lib/
ls
cd sound_play/
ls
vi say.py
cp /opt/ros/indigo/lib/sound_play/say.py ~/catkin_ws/src/check_odometry/src/
l
ls ~/catkin_ws/src/check_odometry/src/
cd ~/catkin_ws/src/check_odometry/
ls
cd s
cd src/
ls
vi check.py
vi say.py 
rosrun check_odometry say.py 
rosrun check_odometry say.py "hi"
vi say.py 
vi check.py 
rosrun check_odometry check.py 
rosrun check_odometry check.py "hi"
vi check.py 
rosrun check_odometry check.py "hi"
rosrun check_odometry check.py 
vi say.py
vi check.py 
rosrun check_odometry check.py 
vi check.py
vi say.py
rosrun check_odometry check.py 
vi say.py
rosrun check_odometry check.py 
roslaunch check_odometry check_odom.launch 
vii say.py
vi say.py
roslaunch check_odometry check_odom.launch 
rosrun soudn
rosrun sound_play say.py "hi"
roslaunch check_odometry check_odom.launch 
vi check.py
roslaunch check_odometry check_odom.launch 
vi check.py
cd catkin_ws/src/
cd check_odometry/
ls
cd launch/
sudo vi check_odom.launch 
cd ../src/
ls
sudo vi check.py 
roslaunch check_odometry check_odom
roslaunch check_odometry check_odom.launch
sudo vi check.py 
roslaunch check_odometry check_odom.launch
sudo vi check.py 
roslaunch check_odometry check_odom.launch
sudo vi check.py 
ls
roslaunch check_odometry check_odom.launch

rosrun sound_play say.py 'ya'
rosrun sound_play say.py 'mask reul saw joo sae yo'
yesman
good
goo
ls
cd /opt/ros/indigo/share/sound_play/
ls
cd sounds/
ls
cd ..
ls
cd msg/
ls
cd ../../..
cd lib/
cd sound_play/
ls
rosrun sound_play play.py
sudo vi play.py 
roslaunch rosbridge_server rosbridge_websocket.launch 
roslaunch sound_play soundplay_node.launch
rosrun sound_play play.py
rosrun sound_play say.py 'ya'
rosrun sound_play play.py --help
rosrun sound_play play.py say-beep.wav

rosrun sound_play play.py say-beep.wav
rosrun sound_play play.py KMS.wav
rosrun sound_play shutup.py
rosrun sound_play say.py 'mask reull'
rosrun sound_play say.py 'mask reul'
rosrun sound_play say.py 'mask reul sseo'
rosrun sound_play say.py 'mask reul saw joo sae yo'
cd catkin_ws/src/
ls
rosrun check_odometry say.py
sudo vi say.py
ls
cd check_odometry/
ls
cd src/
ls
sudo vi say.py
rosrun check_odometry say.py
sudo vi say.py
rosrun check_odometry say.py
rosrun check_odometry check.py
cd /opt/ros/indigo/lib/sound_play/
ls
sudo vi say.py 
cd ~/catkin_ws/src/
ls
cd check_odometry/
ls
cd src/
ls
sudo vi say.py
ls
rosrun check_odometry say.py
sudo vi say.py
rosrun check_odometry say.py
rosrun check_odometry say.py 'ya'
sudo vi check.py
cd catkin_ws/src/check_odometry/src/
ls
ls -a
ls
sudo vi say.py
rosrun check_odometry check.py 
vi check
vi check.py
rosrun check_odometry check.py 
~/catkin_ws/src/check_odometry/src/
l
cd ~/catkin_ws/src/check_odometry/src/
l
vi say.py
rosrun check_odometry check.py 
vi check.py
rosrun check_odometry check.py 
vi check.py
rosrun check_odometry check.py 
vi check.py
ifififififififif
cd catkin_ws/src/
ls
cd patrolling/
ls
cd src/
ls
sudo vi patrol.py
cpp
cd ../..
ls
cd robot_local_control/
s
ls
cd msg/
ls
cd ..
cd src/
ls
sudo vi local_control_navigation.cpp 
cd /opt/ros/indigo/lib/
ls
cd sound_play/
l
vi say.py 
cd ~/catkin_ws/src/
l
cd check_odometry/
l
cd src/
l
vi check.py 
vi say.py
rosrun check_odometry say.py 
vi say.py
vi check.py
vi say.py
rosrun check_odometry say.py 
vi say.py
vi check.py
vi say.py
vi check.py
vi say.py
rosrun check_odometry say.py 
vi check.py
rosrun check_odometry say.py 
vi check.py
vi say.py
rosrun check_odometry say.py 
vi say.py
rosrun check_odometry say.py 
vi say.py
rosrun check_odometry say.py 
vi say.py
vi check.py
rosrun check_odometry say.py 
vi check.py
rosrun check_odometry say.py 
vi check.py
rosrun check_odometry check.py 
vi check.py
rosrun check_odometry check.py 
vi check.py
rosrun check_odometry check.py 
vi say.py
vi check.py
rosrun check_odometry check.py 
vi say.py
rosrun check_odometry say.py 
vi say.py
rosrun check_odometry say.py 
l
vi check.py
rosrun check_odometry check.py 
vi check.py
rosrun check_odometry check.py 
vi check.py
rosrun check_odometry check.py 
vi check.py
clear
cd /
ls
find enable
find enable_software_runstop
roslaunch check_odometry  check_odom.launch 
rostopic echo /odom
cd catkin_ws/src/
ls
cd check_odometry/
ls
cd launch/
ls
sudo vi check_odom.launch 
roslaunch check_odometry check_odom.launch
rosrun check_odometry check.py
cd catkin_ws/src/check_odometry/src/
ls

ls
sudo vi check.py
cd catkin_ws/src/check_odometry/src/
ls
sudo vi say.py
rostopic echo /odom
cd ~/catkin_ws/src/patrolling/src/
l
vi patrol.py
rosrun check_odometry check.py 
vi check.py
cd ~/catkin_ws/src/check_odometry/src/
l
vi check.py
rosrun check_odometry check.py 
vi check.py
rosrun check_odometry check.py 
vi check.py
rosrun check_odometry check.py 
vi check.py
roslaunch check_odometry check_odom.launch 
vi check.py
roslaunch check_odometry check_odom.launch 
vi check.py
rosrun patrolling patrol.py
cd ~/catkin_ws/src/patrolling/
l
cd src/
l
vi patrol.py
rosrun patrolling patrol.py
pip install keyboard
rosrun patrolling patrol.py
rosrun teleop_twist_keyboard teleop_twist_keyboard.py 
rosrun patrolling patrol.py
exit
roslaunch fetch_navigation freight_nav.launch 
rorun patrolling patrol.py
rosrun patrolling patrol.py
vi check.py
cd ~/catkin_ws/src/patrolling/src/
l
vi patrol.py
rosrun patrolling patrol.py
vi check.py
cd ~/catkin_ws/src/patrolling/src/
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
cd ~/catkin_ws/src/patrolling/
cd src/
l
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
roslaunch fetch_navigation freight_nav.launch 
cd catkin_ws/
cd catkin_ws/src/
ls
cd odom_subscriber/
ls
cd src/
ls
sudo vi odom_sub.cpp
cd ../..
ls
cd robot_local_control/
ls
cd src/
ls
cd ../..
ls
cd check_odometry/
ls
cd src/
ls
l
cd ../../..
ls
cd src/
ls
cd patrolling/
ls
cd src/
ls
sudo vi patrol.py
cd catkin_ws/src/check_odometry/
ls
cd src/
ls
touch test.py
ls
python test.py
sudo vi test.py
ls
cd /opt/ros/indigo/lib/
ls
cd fetch_navigation/
ls
cd ../..
cd share/fetch_navigation/
ls
cd launch/
ls
sudo vi freight_nav.launch 
l
roslaunch fetch_navigation freight_nav.launch
cd ..
ls
cd launch/
ls
roslaunch fetch_navigation freight_nav.launch
sudo vi freight_nav.launch 
roslaunch fetch_navigation freight_nav.launch
cd catkin_ws/src/patrolling/
cd src/
ls
sudo vi patrol.py
pip install request
roslaunch fetch_navigation freight_nav.launch 
sudo vi test.py 
cd /opt/ros/indigo/share/fetch_navigation/
cd launch/
ls
sudo vi freight_nav.launch 
roslaunch fetch_navigation freight_nav.launch 
cd ~
sudo vi test.py 
python test.py 
roslaunch fetch_navigation freight_nav.launch 
cd /opt/ros/indigo/share/fetch_navigation/launch/
ls
sudo vi freight_nav.launch 
roslaunch fetch_navigation freight_nav.launch 
cd ~/catkin_ws/src/patrolling/src
l
vi patrol.py
rosrun patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
sudo vi patrol.py
rosrun patrolling patrol.py
roslaunch robot_local_control robot_local_control.launch 
roslaunch rostful rostful_server.launch 
cd catkin_ws/src/patrolling/
ls
cd src/
ls
cd ../../..
cd src/
cd robot_local_control/
ls
cd src/
ls
cd ..
l
cd ..
l
cd ..
l
cd ..
l
python test.py
vi test.py
python test.py
vi test.py
python test.py
vi test.py
roslaunch fetch_navigation freight_nav.launch 
roslaunch rostful rostful_server.launch 
cd ~/catkin_ws/src/patrolling/src
l
rosrun patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
cd catkin_ws/src/patrolling/src/
ls
sudo vi patrol.py
rosrun patrolling patrol.py
roslaunch robot_local_control robot_local_control.launch 
cd ~/catkin_ws/src/
roslaunch fetch_navigation freight_nav.launch 
rosrun patrolling patrol.py
cd ~/catkin_ws/src/patrolling/
cd src/
l
vi patrol.py
roslaunch fetch_navigation freight_nav.launch 
cd ~/catkin_ws/src/patrolling/src
sudo vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
cd ~/catkin_ws/src/patrolling/src/
sudo vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
python --version
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
roslaunch robot_local_control robot_local_control.launch 
rostopic echo /odom
roslaunch check_odometry check_odom.launch 
rosrun check_odometry say.py 
roslaunch check_odometry check_odom.launch 
roslaunch rostful rostful_server.launch 
yesman
yes
no
y
ya
rosrun check_odometry check.py
ok im ok
ok
o
ok
k
ok
wd
ok
wd
ok
wd
ok
wd
ok
wd
ok
wd
dkow
dkwo
YES
yes
no
No
NO
yes
roslaunch check_odometry check_odom.launch 
roslaunch fetch_navigation freight_nav.launch 
roslaunch robot_local_control robot_local_control.launch 
rosrun patrolling patrol.py
roslaunch rostful rostful_server.launch 
cd catkin_ws/src/patrolling/src/
sudo vi patrol.py
roslaunch fetch_navigation freight_nav.launch
rosrun patrolling patrol.py
cd catkin_ws/src/patrolling/src/
sudo vi patrol.py
roslaunch rostful rostful_server.launch 
roslaunch robot_local_control robot_local_control.launch 
roslaunch fetch_navigation freight_nav.launch 
cd catkin_ws/src/patrolling/
ls
cd src/
ls
sudo vi say.py 
cd ~/catkin_ws/src/patrolling/src
l
vi patrol.py
cd catkin_ws/src/
l
cd check_odometry/
l
cd src/
l
vi check.py
cd catkin_ws/src/check_odometry/src/
ls
sudo vi check.py
rosrun patrolling patrol.py
cd catkin_ws/src/check_odometry/src/
l
cd ..
patrolling/
l
cd patrolling/
l
cd src/
l
vi patrol.py
cd ..
vi patrol.py
cd src/
k
l
sudo vi patrol.py
rosrun patrolling patrol.py
sudo vi patrol.py
rosrun patrolling patrol.py
sudo vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
atkin_ws/src/
l
cd check_odometry/
l
cd src/
l
vi check.py
cd catkin_ws/src/check_odometry/src/
ls
sudo vi check.py
rosrun patrolling patrol.py
roslaunch fetch_navigation freight_nav.launch 
cd catkin_ws/src/
ls
cd patrolling/
ls
cd src/
ls
sudo vi say.py 
sudo vi patrol.py
vi check.py
sudo vi patrol.py
rosrun patrolling patrol.py
vi check.py
vi patrol.py
rosrun patrolling patrol.py
roslaunch fetch_navigation freight_nav.launch 
vi patrol.py
cd catkin_ws/src/patrolling/src/
vi patrol.py
sudo vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
sudo vi patrol.py
rosrun patrolling patrol.py
zip -r hello.zip ~/home
zip -r hello.zip . -i /home/fetch/home
tar -cvf hello.tar /home/fetch/home
tar -cvf hello.tar ~/home/fetch/home
tar -cvf hello.tar ~/home/fetch/home/
tar -cvf hello.tar ~/home
tar -cvf hello.tar /home
sudo tar -cvf hello.tar /home
ls
sudo tar -cvf hello2.tar /opt
roslaunch fetch_navigation freight_nav.launch 
rosrun patrolling patrol.py
exit
cd C:\Users\hsw08\Desktop\robot
exit
roslaunch fetch_navigation freight_nav.launch 
rosrun patrolling patrol.py
cd ~/catkin_ws/src/patrolling/src
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
cd ~/catkin_ws/src/patrolling/src
l
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
cd ~/catkin_ws/src/patrolling/src/
l
vi patrol.py
rosrun patrolling patrol.py
roslaunch fetch_navigation freight_nav.launch 
cd ~/catkin_ws/src/patrolling/src/
l
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rm .patrol.swp
vi patrol.py
rm .patrol.py.swp
rm .patrol.swp
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py

rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
roslaunch rosbridge_server rosbridge_websocket.launch
l
ls -a
vi ~/.bash
vi bashrc
vi ~/.bashrc
exit
exit
roslaunch rosbridge_server rosbridge_websocket.launch
exit
cd ~/catkin_ws/src/patrolling/s
cd ~/catkin_ws/src/patrolling/src/
l
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rostopice echo /odom
rostpic echo /odom
rostopic echo /odom
rostopic echo /amcl_pose
roslaunch check_odometry check_odom.launch 
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun teleop_twist_keyboard teleop_twist_keyboard.py 
roslaunch fetch_navigation freight_nav.launch 
cd catkin_ws/src/
ls
git clone https://github.com/osrf/rvizweb
ls
curl -sL https://deb.nodesource.com/setup_8.x | sudo bash -
cd ..
rosdep install --from-paths src --ignore-src -r -y
catkin_make intall
roslaunch fetch_navigation freight_nav.launch 
rosrun patrolling patrol.py 
cd ~/catkin_ws/src/patrolling/src
l
vi patrol.py
rosrun patrolling patrol.py 
vi patrol.py
rosrun patrolling patrol.py 
vi patrol.py
rosrun patrolling patrol.py 
eixt
exit
cd ~/catkin_ws/src/patrolling/src/

vi patrol.py
rosrun patrolling patrol.py
vi patrol.py
rosrun patrolling patrol.py
rosrun check_odometry say.py 
rosrun check_odometry say.py "aa"
roslaunch check_odometry check_odom.launch 
vi ~/catkin_ws/src/check_odometry/src/
cd ~/catkin_ws/src/check_odometry/src
l
vi check.py
vi say.py
rosrun check_odometry say.py 
roslaunch fetch_navigation freight_nav.launch 
roslaunch fetch_navigation freight_nav.launch 
roslaunch fetch_navigation freight_nav.launch 
vi ~/.bashrc
ls
rosrun sound_play say.py "hihihi"
roslaunch fetch_navigation freight_nav.launch 
cd ~/ws/src
l
cd catkin_ws/
rosdep init
cd src/
l
history
cd ~
l
ls -a
exit
rostopic list
gazebo
sudo apt-get install -y ros-indigo-pr2-gazebo
ls
roslaunch pr2_gazebo pr2_empty_world.launch
ls
source install/setup.bash
roslaunch rvizweb rvizweb.launch
ls
cd src/
ls
ls
cd catkin_ws/
ls
roslaunch rvizweb rvizweb.launch
cd install/
ls
cd share/
ls
cd ..
ls
cd ..
ls
sudo rm -rf install/
ls
catkin_make
ls
cd src/
ls
sudo rm -rf rvizweb
ls
cd ..
ls
cd build/
ls
cd ..
catkin_make
ls
cd build/
ls
sudo rm -rf rvizweb/
ls
cd ..
ls
cd devel/
ls
sudo vi setup.bash 
cd ../
catkin_make
cd src/
ls
rosrun check_odometry say.py
cd check_odometry/
ls
cd launch/
ls
roslaunch check_odometry check_odom.launch 
cd ..
ls
cd src/
ls
rosrun check_odometry say.py 
cd ../..
cd ..
ls
ls -a
rosdep install --from-paths src --ignore-src -r -y
ls
catkin_make
catkin_make install
ls
cd install/
ls
sudo vi setup.bash 
cd ..
ls
sudo -rm -rf i
sudo rm -rf install
;s
ls
l
cd ..
ls
cd ..
ls
cd fetch/
ls
cd catkin_ws/
ls
cd src/
ls
cd odom_subscriber/
ls
cd src/
ls
cd ../..
ls
cd..
cd ..
catkin_make
cd ..
sudo vi .bashrc 
rosrun check_odometry say.py 
rosrun sound_play say.py 
rosrun sound_play say.py "ya"
roscore
cd ~/catkin_ws/
ls -a
sudo vi .catkin_workspace 
ls
cd devel/
ls
sudo vi setup.bash 
sudo vi setup.s
sudo vi setup.ss
sudo vi setup.sh
sudo vi env.sh 
ls
sudo vi _setup_util.py 
ls
rosrun sound_play say.py
rosrun sound_play say.py "ya"
vi ~/.bashrc
rosrun check_odometry say.py 
rosrun sound_play say.py "hi"
cd catkin_ws/
catkin_make install
ls
cd install/
ls
cd ..
. ~/ws/install/setup.bash
. ~/catkin_ws/install/setup.bash
roslaunch rvizweb rvizweb.launch
ls
cd src/
ls
cd rvizweb/
ls
cd docker/
ls
cd ..
ls
cd launch/
ls
sudo vi rvizweb.launch 
cd ..
ls
sudo vi index.html
ls
sudo vi index.html 
curl --help
ls
cd ..
ls
cd rvizweb
ls
roslaunch fetch_navigation freight_nav.launch 
cd ~/catkin_ws/
cd src/patrolling/src/
l
vi patrol.py
rosrun patrolling patrol.py
rosrun sound_play say.py "hhihihihiihhi"
rosrun patrolling patrol.py
cd ~/catkin_ws/src/patrolling/src/
l
vi patrol.py
roslaunch fetch_navigation freight_nav.launch 
rostopic echo /amcl_pose
roslaunch fetch_navigation freight_nav.launch 
rosrun map_server map_saver -f capture
roslaunch fetch_navigation build_map.launch 
rosrun patrolling patrol.py
cd catkin_ws/src/patrolling/src/
l
vi patrol.py
rosrun patrolling patrol.py
roslaunch fetch_navigation freight_nav.launch 
rosrun patrolling patrol.py
cd ~/catkin_ws/src/patrolling/src/
l
vi patrol.py
rosrun patrolling patrol.py
roslaunch fetch_navigation freight_nav.launch 
cd catkin_ws/src/patrolling/src/
l
vi patrol.py
roslaunch fetch_navigation freight_nav.launch 
rosrun patrolling patrol.py
rostopic echo /amcl
rostopic echo /amcl_pose
rostopic echo /odom
rostopic echo /amcl_pose
roslaunch fetch_navigation freight_nav.launch 
cd catkin_ws/src/
l
vi check_odometry/
cd check_odometry/
l
cd src/
l
cd ..
l
cd patrolling
l
cd ..
cd check_odometry/
l
cd src/
l
vi check.py
roslaunch rosbridge_server robridge_websocket.launch
cd catkin_ws/
ls
cd src/
ls
cd /opt/ros/indigo/share/
ls
cd rosbridge_server
ls

ls
roslaunch rosbridge_server rosbridge_websocket.launch
ls
cd /opt/ros/indigo/
ls
cd lib/
ls
cd amcl/
ls
sudo vi amcl 
cd ..
ls
cd rviz/
ls
sudo vi rviz 
cd ..
cd share/
ls
cd rviz
ls
cd icons/
ls
cd ..
ls
cd ogre_media/
ls
cd ..
ls
cd ..
cd rqt
ls


roslaunch fetch_navigation freight_nav.launch
roslaunch fetch_navigation freight_nav.launch
roscore
exit
rostopic list
ls
cd
cd /
ls
cd home/fetch
ls
roscore
rosrun map_server map_server /home/fetch/final_map.pgm
roslaunch map_server map_server /home/fetch/final_map.pgm
rosrun map_server map_server ~/final_map.pgm
cd catkin_ws/src/rosbridge_suite/
ls
cd rosbridge_server/
ls
cd launch/
ls
sudo vi rosbridge_websocket.launch 
rostopic list
cd /opt/ros/indigo/
ls
cd include/
ls
cd ..
ls
cd bin/
ls
cd rostopic
sudo vi rostopic 
cd ..
ls
cd lib/
ls
cd rostopic
ls
sudo vi rostopic 
cd ..
cd rviz
ls
sudo vi rviz 
cd ..
ls
cd web_video_server/
ls
cd ..
ls
cd share/
ls
cd rostopic/
ls
cd cmake/
ls
sudo vi rostopicConfig.cmake 
cd ../
cd ..
cd rviz/
ls
rosrun rviz rviz
roslaunch rosbridge_server rosbridge_websocket.launch
ls
cd catkin_ws/
ls
cd src/
ls
cd patrolling/
ls
cd src/
ls
sudo vi patrol.py
roslaunch gaitech_remote_control remote_control.launch 
rosrun teleop_twist_keyboard teleop_twist_keyboard.py 
cd catkin_ws/src/patrolling/
cd ~
tar -cvf patrolling.tar catkin_ws/src/patrolling/
l
cd /opt/ros/indigo/
ls
cd share/
ls
cd sound_play/
ls
cd ..
cd festival
cd sound_play/
ls
sudo vi soundplay_node.launch 
rosrun sound_play say.py
rostopic echo /odom
exit
cd catkin_ws/src/patrolling/src/
l
vi patrol.py
ls
cd sounds/
ls
cd ..
ls
cd action/
ls
cd ..
cd lib/
ls
cd sound_play/
ls
sudo vi soundplay_node.py 
sudo vi say.py
rosrun sound_play say.py "ya"
sudo vi say.py
rosrun sound_play say.py "ya"
rostopic echo /odom
q!
cd catkin_ws/src/check_odometry/src/
l
vi check.py
python check.py
vi check.py
python check.py
vi check.py
python check.py
vi check.py
python check.py
vi check.py
python check.py
vi check.py
python check.py
vi check.py
python check.py
cd //
cd ~
cd ..
cd ~
cd catkin_ws/src/patrolling/src/
l
vi patrol.py
cd ..
cd src
l
cd check_odometry/
l
cd src/
l

vi check.py
python check.py
vi check.py
python check.py
vi check.py
python check.py
vi check.py
python check.py
vi check.py
python check.py
vi check.py
python check.py
vi check.py

vi check.py
rm .check.py.swp
vi check.py
python check.py
vi check.py
cd catkin_ws/src/patrolling/
cd src/
l
vi patrol.py
cd ..
l
cd check_odometry/
l
cd launch/
l
cd ..
l
cd src/
l
vi check.py
python check.py
vi check.py
python check.py
vi check.py
python check.py
vi check.py
python check.py
vi check.py
python check.py
vi check.py
python check.py
vi check.py
python check.py
vi check.py
python check.py
vi check.py
python check.py
vi check.py
python check.py
wq
vi check.py
python check.py
vi check.py
cd catkin_ws/src/patrolling/src/
l
vi patrol.py
rosrun sound_play say.py "ya"
roslaunch fetch_navigation freight_nav.launch 
rostopic 
rostopic info
rostopic info /amcl_pose
rostopic info /odom
rostopic info /sound_play
cd catkin_ws/src/patrolling/src/
vi patrol.py
roslaunch rosbridge_server rosbridge_websocket.launch 
rostopic find /amchl_pose
rostopic find /amcl_pose
rostopic bw
rostpic bw /amcl_pose
rostopic bw /amcl_pose
rostopic find /acml_pose
rostopic find /odom
cd ~
rostopic find /odom
cd /home
l
rostopic echo /odom
rostopic find /odom
rostopic find /amcl_pose
cd ~
rostopic find /amcl_pose
rosrun teleop_twist_keyboard teleop_twist_keyboard.py 
rostopic echo /amcl_pose
cd catkin_ws/src/patrolling/src
vi patrol.py
rostopic echo /amcl_pose
cd /opt/ros/indigo/lib/
ls
cd sound_play/
ls
sudo vi say.py 
cd /opt/ros/indigo/share/
ls
cd web_video_server/
ls
cd cmake/
ls
cd ../..
cd ..
cd lib/
ls
cd web_video_server/
ls
rostopic list
roslaunch rosbridge_server rosbridge_websocket.launch
rostopic list
cd ..
rostopic | grep sound
rostopic / grep sound
rostopic echo/ grep sound
rostopic echo /grep sound
cd rostopic
ls
rostopic lsit
rostopic list
cd ../
rostopic info
rostopic info /amcl_pose
rostopic info /odom
rostopic info /sound_play
rostopic type /sound_play
rostopic type /sound_play/goal
rostopic info /sound_play/goal
cd sound_play/
ls
sudo vi say.py
cd ../../share/
cd sound_play/
ls
cd action/
ls
cd ..
cd msg/
ls
cd ..
ls
sudo vi test.launch 
sudo vi soundplay_node.launch 
ls
cd cmake/
ls
cd ..
ls
cd action/
ls
sudo vi SoundRequest.action 
cd ..
ls
cd msg/
ls
cd ../../../..
find SoundClient
find libsoundplay
ls
cd indigo/
ls
cd lib/
ls
cd sound_play/
ls
sudo vi test
sudo vi test_actionlib_client.py 
ls
sudo vi soundplay_node.py 
rostopic list
rostopic info /sound_play/
rostopic info /sound_play
rostopic info /sound_play/result
rostopic info /sound_play/feedback
rostopic list
rostopic info /robot_state
rostopic info /sound_play/status
roslaunch rosbridge_server rosbridge_websocket.launch
roslaunch rosbridge_server rosbridge_websocket.launch 
roslaunch check_odometry check_odom.launch 
cd catkin_ws/src/check_odometry
ls
cd src/
python check.py
vi check.py
python check.py
