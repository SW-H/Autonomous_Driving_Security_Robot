## ROS 관련 코드 설명

```python
class Patrol:
 
    def __init__(self):
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()
 
    def set_goal_to_point(self, point):
        # 관련 클래스 로드
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now() #현재시간 저장 
        # 목표 지점 x,y 좌표 지정
        goal.target_pose.pose.position.x = point[0]
        goal.target_pose.pose.position.y = point[1]
        # 목표 지점 방향 지정
        quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0, point[2])
        goal.target_pose.pose.orientation.x = quaternion[0]
        goal.target_pose.pose.orientation.y = quaternion[1]
        goal.target_pose.pose.orientation.z = quaternion[2]
        goal.target_pose.pose.orientation.w = quaternion[3]
        # 목표 지점으로 이동 명령
        self.client.send_goal(goal)
        wait = self.client.wait_for_result()
        if not wait:
            # 에러메시지 출력 (서버연결에 문제가 있는 경우)
            rospy.logerr("Action server not available!")
            rospy.signal_shutdown("Action server not available!")
        else:
            return self.client.get_result()
```
   ↳ move_base node를 기반으로 로봇을 목표지점으로 작동시키는 클래스 정의   

```python
   if __name__ == '__main__':
    rospy.init_node('patrolling')
    try:
        p = Patrol()
        while not rospy.is_shutdown():
            # 지정한 지점들을 순찰 ( 목표지점을 반복적으로 지정 )
            if data.decode()=="":
                rospy.loginfo("Sending Waypoint %d - %s", i, w[0])
                rospy.init_node('')
                p.set_goal_to_point(w[1])
            else:
                sound_play_func() # TTS  (음성 안내 출력)
                rospy.loginfo("Sending Waypoint %d - %s", i, w[0]) # 로그 출력
                rospy.init_node('')
                # (AI모델 결과에 따라) 수정된 목표 지점으로 이동
                p.set_goal_to_point(edited_goal)
                print(edited_goal)
 
    except rospy.ROSInterruptException:
        rospy.logerr("Something went wrong when sending the waypoints")
  ```
  ↳ 로봇이 종료될 때까지 입력된 지점들로 반복적으로 로봇을 순회시키는 반복문
  
  
