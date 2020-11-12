# 다중이용시설 마스크 착용 감시 자율 주행 로봇

-------------------------------------------------------------------------------------------------------------------------


## 개요
> ### 자율주행 로봇의 필요성
> 1. 한 자리에 머물지 않기 때문에 사각지대가 존재하는 CCTV의 단점을 보완할 수 있다.
> 2. 실시간 위치정보와 촬영 데이터를 시각화하여 관리자가 쉽게 확인할 수 있다.
> 3. 인력을 로봇으로 대체함으로써 코로나 확산 가능성을 최소화하고 인력 낭비를 방지할 수 있다.
> 4. 360도 카메라와 인공지능을 활용하여 마스크를 착용하지 않은 사람을 효율적으로 찾아내고 음성경고 및 기록을 할 수 있다.


> ### 목적
> 1. 자율주행 로봇이 정해진 루트를 순찰하면서 시설의 실시간 이미지를 촬영한다.
>
>        - 사전에 제작된 지도 위에 입력된 목표 지점들을 순서대로 반복하여 주행한다.
>        - 순찰 중 등장하는 장애물들을 감지하고 피한다.
>
> 2. 자율주행 로봇은 다음의 역할들을 수행한다.
>
>         - 시설 이용객의 마스크 착용 여부를 감지한다.
>         - 마스크 미착용자를 추적 후 경고 멘트를 송출하고 관리자에게 정보를 전달한다.
>         - 마스크 미착용자의 신원을 확보한다.
>
> 3. 웹을 통해 관리자가 촬영 화면 및 detection 결과를 볼 수 있으며, 필요 시 경고에 대한 알림을 받을 수 있다.
>
>         - Admin Web에서는 촬영 이미지와 detection결과가 나타난 이미지를 확인할 수 있다.
>         - Web server는 local 환경에서 구축하며 배포하지 않는다.
>         - 관리자는 로그인을 통해 Admin Web을 이용할 수 있다.
>
> 4. 유사시에 관리자가 로봇을 비상 작동 시킬 수 있다.
>
>         - Admin Web에서 관리자가 로봇을 강제로 조작할 수 있다.
>         - 필요 시 관리자의 메시지를 로봇에 전달하여 음성을 출력할 수 있다.

>
> ### 기대효과
> 1. 코로나 예방 수칙 관리 체계의 효율성 증대
> 2. 지속적인 데이터 수집과 로봇 규제 개선에 따른 감시 성능 향상 기대
> 3. 시간의 제약을 받지 않으므로 필요 경비 인력 감소
> 4. 수집된 데이터를 바탕으로 여러 방면으로 활용 가능

----------------------------------------------------------------------------------------------------

## 개발환경

+ 파이썬 스펙

        Python==3.6.7
        Cython
        dlib==19.8.1
        face-recognition==1.3.0
        numpy==1.15.0
        torch==1.7.0+cu101
        torchvision==0.8.1+cu101
        tensorflow==1.13.1
        tensorflow-gpu==1.13.1
        tqdm
        Keras==2.3.1
        opencv-python
        scikit-learn==0.21.2
        scipy==1.4.1
        Pillow
        visdom
        Nibabel
        GTTS


+ 노트북 스펙

        GPU : GTX 1650 ti
        CPU : i7-10750H
        RAM : 16GB
        CUDA 10.0
        Cudnn : 7.5
        Nvidia driver
        Visual Studio 14 (2015)
        
+ ROS 환경

        Ubuntu 14.04.5
        ROS-indigo 
        
+ 서버 환경 & 네트워크 & DB

        Flask
        Python socket(UDP, TCP)
        SQLite

---------------------------------------------------------------------------------------------------------------------

## 로봇 하드웨어
<img src="/README_img/로봇하드웨어.PNG" width="60%" height="60%" title="로봇하드웨어" alt="로봇하드웨어"></img>    
 
---------------------------------------------------------------------------------------------------------------------

## 전체 시스템 디자인 
![Sytem_Design](/README_img/시스템디자인.PNG "시스템디자인")

---------------------------------------------------------------------------------------------------------------------

## 전체 프로세스 알고리즘
![전체프로세스](/README_img/전체프로세스.PNG "전체프로세스")

---------------------------------------------------------------------------------------------------------------------

## 파트 별 설명    
 1. [Panorama Camera](https://github.com/SW-H/Autonomous_Driving_Security_Robot/blob/main/README_hyperlink/PanoramaCamera.md)
 ---------------------------------------------------------------------------------------------------------------------
 2. AI model 
> 로봇에 장착된 카메라를 통해 수집된 이미지에서 목표한 기획에 맞게끔 자율주행 로봇의 움직임을 결정할 데이터를 도출하기 위해 다음과 같은 인공지능 모델들을 사용하였다.    
>  
>> + Mask Detection (YOLO v4) – Custom Data      
   파노라마 카메라로 수집한 이미지에서 마스크를 쓴 사람과 안 쓴 사람, 잘못 쓴 사람의 얼굴을 detection해내기 위한 CNN모델이다.   Kaggle에서 제공하는 VOC format의 Mask Detection Dataset을 convert2Yolo 툴을 이용해  YOLO에 맞는 데이터 형식으로 변환 후, Google Colab Pro 환경에서 직접 모델을 train시켜 weights값을 생성하였다.   이미지에서 마스크를 쓴 얼굴(with_mask), 마스크를 쓰지 않은 얼굴( without_mask), 마스크를 제대로 쓰지 않은 얼굴(mask_weared_incorrect)을 찾아낸다.               
   ![model_training](/README_img/model_training.PNG "model_training")    ↳ Colab Pro에서 진행한 model training이 완료된 화면과 이에 사용한 parameter   
   ![코드 실행 시 마스크 착용 여부에 따라 구분된 모습](/README_img/detecting_mask_nomask.PNG "코드 실행 시 마스크 착용 여부에 따라 구분된 모습")   ↳코드 실행 시 마스크 착용 여부에 따라 구분된 모습
>>
>>
>>
>>
>> + Person Detection (YOLO v4) – Coco Dataset   
    Mask detection model만으로는 사람의 뒷모습을 잡아내지 못하여 한번 포착한 마스크 미착용자를 지속적으로 tracking할수가 없다. 따라서 사진 촬영 각도에 상관없이 이미지에서 사람을 detection 해낼 필요가 있었다.   
	 Detection 성능의 향상을 위해 Mask detection과 별개의 모델을 사용하였으며, coco dataset으로 훈련된 모델에서 ‘person’  label만을 사용하였다. ![detection_result](/README_img/detection_result.PNG "Coco dataset을 이용해 train한 모델의 detection 결과 예시
")    ↳Coco dataset을 이용해 train한 모델의 detection 결과 예시
>>
>>
>>
>>
>> + Object Tracking (Deep-SORT) – Pretrained Model   
   앞에서 detection한 person의  bounding box를 tracking하는 모델이다. 수집된 이미지에서 person마다 각각의  label(track id)을 붙이고 tracking하기 위해 사용한다.![ObjectTracking](/README_img/ObjectTracking.PNG "Real-time으로 person detection & tracking 하는 모델 출력 예시")   ↳ Real-time으로 person detection & tracking 하는 모델 출력 예시
   사용하는 자율주행 로봇 및 카메라의 특성을 고려하여, 연속적으로 촬영한 이미지에서의 원활한 tracking을 위해 model의 hyper parameter들을 조정하였다.   (max_iou_distance = 0.7, max_cos_distance = 0.2)
>>
>>
>>
>>
>> + Face Recognition (dlib + face_recognition)   
   촬영된 이미지에서 Detection된 face를 database에 저장된 face들과 비교해서 개개인을 식별하고 등록되지 않은 face(unknown)를 색출하기 위해 사용한다. 계속해서 업데이트 되고있는 face_recognition api를 사용하며, 이는 전세계 사람들의 얼굴 데이터인 Labeled Faces in the Wild를 기준으로 99.38%의 정확도를 기록하였다.   ![dlib_and_face_recognition](/README_img/dlib_and_face_recognition.PNG "Face Recognition model 사용 예시")   ↳ Face Recognition model 사용 예시
>>
>>
>>
>>
>> + [PC1 Model Code](https://github.com/SW-H/Autonomous_Driving_Security_Robot/blob/main/README_hyperlink/PC1model_code.md)
>> 		+ [모델](https://github.com/SW-H/Autonomous_Driving_Security_Robot/tree/main/pc1_model)
 ---------------------------------------------------------------------------------------------------------------------
 3. ROS
>>
>> + 초기 세팅     
원격 제어를 위해 로봇과 같은 작업 환경 세팅을 위해 Ubuntu 14.04.5 와 ROS-indigo 설치 후 무선 인터넷을 이용하여 연결한다.   로봇내에 기존에는 OS ( ROS )만 설치되어 있었기에 프로젝트 내의 기능 구현을 위해 추가적으로 다양한 패키지 및 라이브러리가 필요했다. 그 목록은 다음과 같다.
>>> 		- actionlib, actionlib_msgs : 로봇의 순찰(patrolling) 기능 구현을 위해 필요하다. 정해진 범위 내에서 반복적으로 이동하고 이벤트 발생 시 제어(스케쥴링)와 운용에 필요한 메시지를 주고 받기 위해 필요한 패키지이다.   
>>>		- rosbridge  : ROS와 non-ROS간의 통신을 위한 패키지이다. 
>>>		- rospy : ROS는 C++로 이루어진 OS로, 이를 파이썬으로 활용하기 위한 패키지이다.
>>>		- sound_play : String을 입력해서 TTS를 실행시키기 위한 패키지이다.
>>>		- AMCL(Adaptive Monte Calro Localization) : 확률 기반으로 로봇 위치 파악을 위한 패키지이다.
>>>		- Base_local_planner : 평면상에서 로봇의 이동 궤적을 결정하고 구동하는 컨트롤러 패키지 이다. 
>>>		- rostopic : 현재 로봇에서 발행되는 데이터 (실시간 위치인 odometry데이터 등 ) 을 확인하기 위해 사용한다. 
>>>		- costmap_2d : 매핑 및 cost map 생성에 사용한다.
>>>		- fetch_navigation : rostopic으로 주행 거리 및 목표 지점 등을 가져와 주행명령을 출력한다.
>>>		- map_server : 3D모델의 2D평면도를 RVIZ에 출력하여 로봇의 이동 경로 등을 확인할 수 있다.
>>>		- Move_base :경로 계획, 장애물 회피, 로봇 주행 제어 등을 위한 노드를 추가한다.
>>>		- PCL(Point Cloud Library) : 다차원 포인트들을 나타내는데 사용되는 데이터 구조로 일반적으로 3차원 데이터를 나타내는데에 사용된다.
>>
>>
>> +  ROS 3D 시각화 툴 ( Rviz )    
로봇과 연동하여 로봇의 위치를 지도 상에서 시각화하고 로봇의 센서 정보, 이동 경로, 레이저 데이터 등을 표시한다. 다양한 기능 구현 및 작업 과정에서의 시각화를 위해 해당 도구를 사용한다.   ![rviz](/README_img/fetch_navigation%20rviz.PNG) ↳ 매핑된 지도를 불러와 Rviz상에 띄운 실행화면 
>>  
>> + 맵핑 ( mapping )    
로봇이 주행할 공간에 대한 지도를 만들기 위한 과정이다. 내부에서 사용될 맵의 정보를 담고 있는 yaml 파일과 이미지 파일인 pgm 파일로 저장된다.   <img src="https://github.com/SW-H/Autonomous_Driving_Security_Robot/blob/main/README_img/build_map.PNG" title="build_map" alt="build_map">    
↳ fetch_navigation 라이브러리 활용한 지도 매핑 과정 캡처 화면  ![map_and_costmap](/README_img/map_and_costmap.PNG)
>> +  ROS Navigation   
   관련 패키지에는 지도 작성을 위한 노드와 자율 주행을 위한 amcl, move_base 노드가 포함되어 있다.  amcl을 이용해 로봇의 위치를 지도 상에서 인식하고 원격으로 로봇을 조종한다. 이 때는 자동으로 장애물을 인식하여 피할 수 있도록 한다.
>> +  Patrolling   지정해준 범위내에서 반복적으로 순찰(patrolling)하고 도중에 나타나는 장애물은 2D Laser Sensor로 감지한다. 자율 주행 중에 마스크 미착용자 확인 등 여러 기능을 수행한다.   <img src="https://github.com/SW-H/Autonomous_Driving_Security_Robot/blob/main/README_img/patrolling.PNG" width="70%" height="70%" title="patrolling" alt="patrolling">  
>> ↳ patrolling 에 필요한 소스 코드 실행 화면      <img src="https://github.com/SW-H/Autonomous_Driving_Security_Robot/blob/main/README_img/patrolling_plan.PNG" width="70%" height="70%" title="patrolling_plan" alt="patrolling_plan">   
>>     ↳ 로봇의 patrolling과정을 맵에 띄운 화면. 이동 경로 계획 등이 포함된다. 
>> + [patrolling 소스코드 설명](https://github.com/SW-H/Autonomous_Driving_Security_Robot/blob/main/README_hyperlink/ROS_code.md)
 ---------------------------------------------------------------------------------------------------------------------


 4. 네트워크 & 서버 & DB
>> + Network : PC와 PC사이, PC와 Robot사이 데이터 전송은 websocket(UDP&TCP)을 사용한다.
>>>	- PC1에서 Main Server로 AI모델의 detection 및 tracking 결과를 TCP로 전송
>>>	- Robot(Freifgt100)에서 Main Server로 로봇의 실시간 위치를 TCP로 전송
>>>	- Main Server에서 Robot(Freight100)으로 로봇이 움직일 위치를 UDP로 전송
>>>	- Main Server에서 Robot(Freight100)으로 로봇이 음성으로 출력할 문구를 TCP로 전송
>> + Server
>>>  	- Main Server는 PC2의 host PC 환경에 Flask를 이용해 구축한다.
>>> 	- Flask에서 Main Server와 별도로 모델의 결과, 로봇 이동 명령, 로봇 위치 수신, 메시지 전송을 위한 별도의 서버를 운영한다.
>> + Database
>>>	- SQLite을 이용해 직원정보와 발생한 이슈에 대한 테이블을 생성한다.
>>>	- Known people의 사진과 알고리즘에서 전송된 사진은 Directory에 저장한다.
 ---------------------------------------------------------------------------------------------------------------------
 5. TTS
>>  경고메시지와 관리자의 메시지를 출력하기 위한 TTS API로 ROS sound_play 패키지를 사용하였다.
```python
rospy.init_node('say', anonymous=True)
soundhandle = SoundClient()
rospy.sleep(1)

voice = 'voice_kal_diphone'
volume = 1.0

if len(sys.argv) == 1:
    s = sys.stdin.read()
    else:
	s = sys.argv[1]

	if len(sys.argv) > 2:
	    voice = sys.argv[2]
	if len(sys.argv) > 3:
	    volume = float(sys.argv[3])

    rospy.loginfo('Saying: %s' % s)
    rospy.loginfo('Voice: %s' % voice)
    rospy.loginfo('Volume: %s' % volume)

    soundhandle.say(s, voice, volume)
```
↳ ROS에서 sound_play를 실행시키는 say.py 코드의 실행부분

 ---------------------------------------------------------------------------------------------------------------------


 6. [알고리즘 코드 ](https://github.com/SW-H/Autonomous_Driving_Security_Robot/blob/main/README_hyperlink/algorithm_code.md)
```python
#target selection
if state == 0:
    #nearest unmasked
    nearest_unmasked = []
    max_area = 0
    for mask in masks:
        if (mask[2] - mask[0]) * (mask[3] - mask[1]) >= max_area and mask[5]!=0:
            nearest_unmasked = mask
            max_area = (mask[2] - mask[0]) * (mask[3] - mask[1])
    #person match
    target = []
    target_id = -1
    if nearest_unmasked != [] and persons != []:
        for person in persons:
            if person[0] <= nearest_unmasked[0] and person[1] <= nearest_unmasked[1] and person[2] >= nearest_unmasked[2] and person[3] >= nearest_unmasked[3] and person[6] not in targets:
                target = person
                target_id = person[6]
                targets.append(target_id)
                state = 1
                img1_grab = np.array(ImageGrab.grab(square))
                img1_cvt = cv2.cvtColor(img1_grab, cv2.COLOR_BGR2RGB)
                break
```
↳ 주행중 마스크를 쓰지 않거나 부정확하게 쓴 인물을 발견하면 타겟으로 설정하고 촬영 사진을 저장한 후 인물을 트래킹하기 시작한다.
```python
if state == 1:
    found = False
    unmasked = True
    for person in persons:
        if person[6] == target_id: #find correct target_id
            theta = ((person[2]+person[0])/2) / (square[2]-square[0]) * math.pi / 180 - math.pi
            if person[3] <= (square[3]-square[1]) * 0.9: #not close enough
                global rob_loc
                rob_loc = [0,0,0]
                z = theta+rob_loc[2]
                x = rob_loc[0] + math.cos(z) * distance
                y = rob_loc[1] + math.cos(z) * distance
                sock.send(str([x,y,z]).encode()) #robot location send
            else: #close enough
                sock.send(str([5]).encode()) #TTS(마스크를 써주세요)
                img2_grab = np.array(ImageGrab.grab(square))
                img2_cvt = cv2.cvtColor(img2_grab, cv2.COLOR_BGR2RGB)
```
↳ 자율주행 로봇이 타겟 인물에 접근하였다고 판단하면 TTS로 음성 경고를 출력한다.
```python
                while delay_time < 10: #wait for 10seconds
                    delay_time = time.time() - TTS_time
 
                    if masked(masks,check): #check if masked or not
                        state = 0
                        unmasked = False
                        sock.send(str([6]).encode()) #TTS(감사합니다)
                        break
                if unmasked: #unmasked until the end
                    state = 0
                    img_name = frame_num
                    cv2.imwrite('./criminal/'+img_name+'.jpg',img2_cvt)
                    face_recognition('./criminal/'+img_name+'.jpg')
```
↳ 경고후 일정 시간 내에 마스크를 착용하면 감시를 멈추고, 마스크를 착용하지 않으면 사진 촬영 후 관리자에게 전송한다.

 ---------------------------------------------------------------------------------------------------------------------
 
 ## 웹 & 서비스
   Flask web frame work로 제작한  ROS 기반 로봇 컨트롤 및 관리 Web page
   다른페이지는 클릭할 시 접속이 가능하지만 유일하게 컨트롤 대시보드를 이용하기 위한 관리자 전용 로그인 기능하다.

 + #### Home & About us
 ![website_home](/README_img/website_home.PNG)
 ↳ about, team에 대한 페이지 전환과 controller, login, join 페이지로의 이동이 가능한 Main page이다.   
    
 ![website_about](/README_img/website_about.PNG)
 ↳ 로봇의 주기능 및 사용된 알고리즘에 대한 설명
    
 ![website_team_introduction](/README_img/website_team_introduction.PNG)
 ↳ 팀 소개
  
 + #### Controller Dashboard
 ![website_team_introduction](/README_img/website_dashboard.PNG)
 ↳ 실시간으로 로봇을 통한 Detection 결과를 영상으로 출력하고 로봇과의 연결상태 확인 및 컨트롤 가능한 page
     
        
	  
 ```python
var app = new Vue({
    el: '#app',
    // storing the state of the page
    data: {
        connected: false,
        ros: null,
        ws_address: 'ws://172.22.77.172:9090',
        logs: [],
        loading: false,
        topic: null,
        message: null,
    },
    
    // helper methods to connect to ROS
    methods: {
        connect: function() {
            this.loading = true
            this.ros = new ROSLIB.Ros({
                url: this.ws_address
            })
            this.ros.on('connection', () => {
                console.log(' - Connected!')
                this.logs.unshift((new Date()).toTimeString() + ' - Connected!')
                this.connected = true
                this.loading = false
            })
            this.ros.on('error', (error) => {
                console.log(` - Error: ${error}`)
                this.logs.unshift((new Date()).toTimeString() + ` - Error: ${error}`)
            })
            this.ros.on('close', () => {
                console.log(' - Disconnected!')
                this.logs.unshift((new Date()).toTimeString() + ' - Disconnected!')
                this.connected = false
                this.loading = false
            })
        },
        disconnect: function() {
            this.ros.close()
        },
```
↳ 웹소켓을 이용하여 js와 ROS를 직접 연결한다.

```python
setTopic: function() {
    this.topic = new ROSLIB.Topic({
        ros: this.ros,
        name: '/cmd_vel',
        messageType: 'geometry_msgs/Twist'
    })
},
forward: function() {
    this.message = new ROSLIB.Message({
        linear: { x: 1, y: 0, z: 0, },
        angular: { x: 0, y: 0, z: 0, },
    })
    this.setTopic()
    this.topic.publish(this.message)
},
stop: function() {
    this.message = new ROSLIB.Message({
        linear: { x: 0, y: 0, z: 0, },
        angular: { x: 0, y: 0, z: 0, },
    })
    this.setTopic()
    this.topic.publish(this.message)
},
backward: function() {
    this.message = new ROSLIB.Message({
        linear: { x: -1, y: 0, z: 0, },
        angular: { x: 0, y: 0, z: 0, },
   })
    this.setTopic()
    this.topic.publish(this.message)
},
turnLeft: function() {
    this.message = new ROSLIB.Message({
       linear: { x: 0, y: 0, z: 0, },
       angular: { x: 0, y: 0, z: 0.5, },
    })
    this.setTopic()
    this.topic.publish(this.message)
},
turnRight: function() {
    this.message = new ROSLIB.Message({
        linear: { x: 0, y: 0, z: 0, },
        angular: { x: 0, y: 0, z: -0.5, },
    })
    this.setTopic()
    this.topic.publish(this.message)
},
```
↳ 웹페이지상에서 관리자가 로봇을 수동으로 조종할 수 있도록 컨트롤 명령을 js를 통하여 ros에 직접 전달한다.
     
      
      
+ ####  Member Management & Authentication administrator
   
```python
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in ', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)
 
)
 
def save_picture(form_picture):
    random_hex=secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static/profile_pics', picture_fn)
    form_picture.save(picture_path)
 
    return picture_fn
 
@app.route("/account", methods=['GET','POST'])
@login_required
def account():
    form =UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash('Your account has been updated!', 'success')
        return redirect(url_for('account'))
    elif request.method=='GET':
        form.username.data=current_user.username
        form.email.data=current_user.email
    image_file = url_for('static', filename='profile_pics/' + current_user.image_file)
    return render_template('account.html', title='Account', image_file=image_file, form = form)
 ```
 ↳ 로봇을 운용하는 기관 직원 및 웹페이지 회원의 정보(사진, 인적사항)을 웹페이지에서 수정하여 바로 Database에 반영한다.
 
```python
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
 
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key = True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    image_file = db.Column(db.String(20), nullable=False, default= 'default.jpg')
    password = db.Column(db.String(60), nullable=False)
    posts = db.relationship('Post', backref='author', lazy=True)
 
    def __repr__(self):
        return f"User('{self.username}','{self.email}','{self.image_file}')"
 
 
class Post(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    title = db.Column(db.String(100), nullable=False)
    date_history = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    content = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable= False)
 
    def __repr__(self):
        return f"Post('{self.title}','{self.date_history}')"
```
 ↳ 신분 확인을 위한 로봇을 운용하는 기관의 직원정보와 로봇이 탐지한 이슈를 database에 저장한다. 
 
```python
@app.route('/model_result')
def model_result():
    global detection_result
    serverSocket = socket(AF_INET, SOCK_STREAM)
    serverSocket.bind((TCP_IP,TCP_PORT))
    serverSocket.listen(0)
    connectionSocket, _ = serverSocket.accept()
    while True:
        data = connectionSocket.recv(8192)
        detection_result = data.decode()
        connectionSocket.send('success'.encode())
        serverSocket.close()
    return "model_result"
	↳ AI Model로부터 detection 및 tracking 결과를 websocket으로 받아오는 model_result 페이지
 
@app.route('/ordering')
def ordering():
```
↳ AI model의 결과 데이터를 기반으로  로봇의 다음 움직임을 알고리즘으로 판단하여  로봇에 websocket으로 전달하는 ordering 페이지, 코드 내용은 위의 알고리즘 코드와 같다.
 
```python
@app.route('/robot_location')
def robot_location():
    global rob_loc
    serverSocket = socket(AF_INET, SOCK_STREAM)
    serverSocket.bind((TCP_IP_LOC,TCP_PORT_LOC))
    serverSocket.listen(0)
    connectionSocket, _ = serverSocket.accept()
   
    while True:
        data = connectionSocket.recv(8192)
        rob_loc = eval(data.decode())
        print(rob_loc)
        connectionSocket.send('success'.encode())
        serverSocket.close()
    return "rob_loc"
```
↳ 저장된 Map 위에서 로봇의 현재 절대좌표를 websocket으로 실시간으로 받아오는 robot_location 페이지
 
```python
@app.route('/video_feed')
def video_feed():
    redirect(url_for('ordering'))
    square = (348,194,1493,660)
    return Response(gen_bboxed_frames(square),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
```
↳ 파노라마 카메라로 촬영한 이미지를 전용 소프트웨어를 거쳐 real-time video형식으로 flask로 가져오는 video_feed 페이지

 ---------------------------------------------------------------------------------------------------------------------
 
 ## 프로젝트 결과

 ![res](/README_img/res.PNG)

------------------------------------------------------------------------------------------------------------------------

## 향후 발전가능성 
 + 공사현장, 의료시설, 실험실, 롤러장 등에서의 안전장비 착용여부 점검(실외라면 오토바이나 자전거)
 + pose estimation을 사용했으므로 위험행동 감지를 통해 경비 기능이 가능
 + 포스트 코로나 시대에도 unknown에 대한 방범 기능으로서 사용 가능
 + 현재 프로세스를 드론에 적용시킨다면 농업 등 다른 산업 분야에 적용 가능

