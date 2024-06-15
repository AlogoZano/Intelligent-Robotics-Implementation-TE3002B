from launch import LaunchDescription
from launch.actions import ExecuteProcess, LogInfo, RegisterEventHandler, DeclareLaunchArgument
from launch.event_handlers import OnProcessStart, OnProcessExit, OnExecutionComplete
from launch.substitutions import FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import TimerAction
import time

def generate_launch_description():
    turtlesim_node = Node(
        package="turtlesim",
        executable="turtlesim_node",
        name="sim",
    )

    turtle_move_node = Node(
        package="ros_move",
        executable="speed_change_node",
        name="sim2",
    )

    bot_move_node = Node(
    package="ros_move",
    executable="bot_node",
    name="sim2",
    )

    spawn_turtle_bot = ExecuteProcess(
        cmd=[
            [
                FindExecutable(name="ros2"),
                " service call ",
                "/spawn ",
                "turtlesim/srv/Spawn ",
                "\"{x: 5.5, y: 1.0, theta: 1.57, name: 'turtlebot'}\"",
            ]
        ],
        shell=True,
    )


    kill_turtle1 = ExecuteProcess(
        cmd=[
            [
                FindExecutable(name="ros2"),
                " service call ",
                "/kill ",
                "turtlesim/srv/Kill ",
                "\"{name: 'turtle1'}\"",
           ]
        ],
        shell=True,
    )

    spawn_turtle_robot = ExecuteProcess(
        cmd=[
            [
                FindExecutable(name="ros2"),
                " service call ",
                "/spawn ",
                "turtlesim/srv/Spawn ",
                "\"{x: 1.0, y: 5.5, theta: 0.0, name: 'turtlerobot'}\"",
            ]
        ],
        shell=True,
    )
    return LaunchDescription(
        [
            turtlesim_node,
            kill_turtle1,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=kill_turtle1,
                    on_exit=[
                        #LogInfo(msg="Turtlesim started, spawning turtle"),
                        spawn_turtle_bot,
                        spawn_turtle_robot          
                    ],
                )
            ),
            bot_move_node,
            turtle_move_node
        ]
    )

    