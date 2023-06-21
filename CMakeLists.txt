project(lookout_transformer)
cmake_minimum_required(VERSION 3.10)

##################################
## Configure CATKIN dependecies ##
##################################
find_package(catkin REQUIRED
   COMPONENTS rospy
)

catkin_python_setup()

###################################
## CATKIN specific configuration ##
###################################
catkin_package(
   CATKIN_DEPENDS rospy
)
