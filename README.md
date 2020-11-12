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
> 1. [Panorama Camera](https://github.com/SW-H/Autonomous_Driving_Security_Robot/blob/main/README_hyperlink/PanoramaCamera.md)
> ---------------------------------------------------------------------------------------------------------------------
> 2. AI model 
>> 로봇에 장착된 카메라를 통해 수집된 이미지에서 목표한 기획에 맞게끔 자율주행 로봇의 움직임을 결정할 데이터를 도출하기 위해 다음과 같은 인공지능 모델들을 사용하였다.    
>>  
>>> + Mask Detection (YOLO v4) – Custom Data      
   파노라마 카메라로 수집한 이미지에서 마스크를 쓴 사람과 안 쓴 사람, 잘못 쓴 사람의 얼굴을 detection해내기 위한 CNN모델이다.   Kaggle에서 제공하는 VOC format의 Mask Detection Dataset을 convert2Yolo 툴을 이용해  YOLO에 맞는 데이터 형식으로 변환 후, Google Colab Pro 환경에서 직접 모델을 train시켜 weights값을 생성하였다.   이미지에서 마스크를 쓴 얼굴(with_mask), 마스크를 쓰지 않은 얼굴( without_mask), 마스크를 제대로 쓰지 않은 얼굴(mask_weared_incorrect)을 찾아낸다.               
   ![model_training](/README_img/model_training.PNG "model_training")    ↳ Colab Pro에서 진행한 model training이 완료된 화면과 이에 사용한 parameter   
   ![코드 실행 시 마스크 착용 여부에 따라 구분된 모습](/README_img/detecting_mask_nomask.PNG "코드 실행 시 마스크 착용 여부에 따라 구분된 모습")   ↳코드 실행 시 마스크 착용 여부에 따라 구분된 모습
>>>
>>>
>>>
>>>
>>> + Person Detection (YOLO v4) – Coco Dataset   
    Mask detection model만으로는 사람의 뒷모습을 잡아내지 못하여 한번 포착한 마스크 미착용자를 지속적으로 tracking할수가 없다. 따라서 사진 촬영 각도에 상관없이 이미지에서 사람을 detection 해낼 필요가 있었다.   
	 Detection 성능의 향상을 위해 Mask detection과 별개의 모델을 사용하였으며, coco dataset으로 훈련된 모델에서 ‘person’  label만을 사용하였다. ![detection_result](/README_img/detection_result.PNG "Coco dataset을 이용해 train한 모델의 detection 결과 예시
")    ↳Coco dataset을 이용해 train한 모델의 detection 결과 예시
>>>
>>>
>>>
>>>
>>> + Object Tracking (Deep-SORT) – Pretrained Model   
   앞에서 detection한 person의  bounding box를 tracking하는 모델이다. 수집된 이미지에서 person마다 각각의  label(track id)을 붙이고 tracking하기 위해 사용한다.![ObjectTracking](/README_img/ObjectTracking.PNG "Real-time으로 person detection & tracking 하는 모델 출력 예시")   ↳ Real-time으로 person detection & tracking 하는 모델 출력 예시
   사용하는 자율주행 로봇 및 카메라의 특성을 고려하여, 연속적으로 촬영한 이미지에서의 원활한 tracking을 위해 model의 hyper parameter들을 조정하였다.   (max_iou_distance = 0.7, max_cos_distance = 0.2)
>>>
>>>
>>>
>>>
>>> + Face Recognition (dlib + face_recognition)   
   촬영된 이미지에서 Detection된 face를 database에 저장된 face들과 비교해서 개개인을 식별하고 등록되지 않은 face(unknown)를 색출하기 위해 사용한다. 계속해서 업데이트 되고있는 face_recognition api를 사용하며, 이는 전세계 사람들의 얼굴 데이터인 Labeled Faces in the Wild를 기준으로 99.38%의 정확도를 기록하였다.   ![dlib_and_face_recognition](/README_img/dlib_and_face_recognition.PNG "Face Recognition model 사용 예시")   ↳ Face Recognition model 사용 예시
>>>
>>>
>>>
>>>
>>> + [PC1 Model Code](https://github.com/SW-H/Autonomous_Driving_Security_Robot/blob/main/README_hyperlink/PC1model_code.md)
> ---------------------------------------------------------------------------------------------------------------------
> 3. ROS
>>>
>>> + 초기 세팅     
원격 제어를 위해 로봇과 같은 작업 환경 세팅을 위해 Ubuntu 14.04.5 와 ROS-indigo 설치 후 무선 인터넷을 이용하여 연결한다.   로봇내에 기존에는 OS ( ROS )만 설치되어 있었기에 프로젝트 내의 기능 구현을 위해 추가적으로 다양한 패키지 및 라이브러리가 필요했다. 그 목록은 다음과 같다.
>>>> 		- actionlib, actionlib_msgs : 로봇의 순찰(patrolling) 기능 구현을 위해 필요하다. 정해진 범위 내에서 반복적으로 이동하고 이벤트 발생 시 제어(스케쥴링)와 운용에 필요한 메시지를 주고 받기 위해 필요한 패키지이다.   
>>>>		- rosbridge  : ROS와 non-ROS간의 통신을 위한 패키지이다. 
>>>>		- rospy : ROS는 C++로 이루어진 OS로, 이를 파이썬으로 활용하기 위한 패키지이다.
>>>>		- sound_play : String을 입력해서 TTS를 실행시키기 위한 패키지이다.
>>>>		- AMCL(Adaptive Monte Calro Localization) : 확률 기반으로 로봇 위치 파악을 위한 패키지이다.
>>>>		- Base_local_planner : 평면상에서 로봇의 이동 궤적을 결정하고 구동하는 컨트롤러 패키지 이다. 
>>>>		- rostopic : 현재 로봇에서 발행되는 데이터 (실시간 위치인 odometry데이터 등 ) 을 확인하기 위해 사용한다. 
>>>>		- costmap_2d : 매핑 및 cost map 생성에 사용한다.
>>>>		- fetch_navigation : rostopic으로 주행 거리 및 목표 지점 등을 가져와 주행명령을 출력한다.
>>>>		- map_server : 3D모델의 2D평면도를 RVIZ에 출력하여 로봇의 이동 경로 등을 확인할 수 있다.
>>>>		- Move_base :경로 계획, 장애물 회피, 로봇 주행 제어 등을 위한 노드를 추가한다.
>>>>		- PCL(Point Cloud Library) : 다차원 포인트들을 나타내는데 사용되는 데이터 구조로 일반적으로 3차원 데이터를 나타내는데에 사용된다.
>>>
>>>
>>> +  ROS 3D 시각화 툴 ( Rviz )    
로봇과 연동하여 로봇의 위치를 지도 상에서 시각화하고 로봇의 센서 정보, 이동 경로, 레이저 데이터 등을 표시한다. 다양한 기능 구현 및 작업 과정에서의 시각화를 위해 해당 도구를 사용한다.   !<img src="/README_img/fetch_navigation%20rviz.PNG" width="70%" height="70%" title="map" alt="map">   ↳ 매핑된 지도를 불러와 Rviz상에 띄운 실행화면 
>>>  
>>> + 맵핑 ( mapping )   
로봇이 주행할 공간에 대한 지도를 만들기 위한 과정이다. 내부에서 사용될 맵의 정보를 담고 있는 yaml 파일과 이미지 파일인 pgm 파일로 저장된다.   [mapping](https://github.com/SW-H/Autonomous_Driving_Security_Robot/blob/main/README_img/build_map.PNG)    ↳ fetch_navigation 라이브러리 활용한 지도 매핑 과정 캡처 화면   
<img src="/README_img/map.PNG" width="50%" height="50%" title="map" alt="map"> </img><img src="/README_img/cost_map.PNG" width="50%" height="50%" title="cost_map" alt="cost_map"></img>      ↳매핑 완료된 맵의 pgm파일	↳ 로봇의 활동 가능 지역을 보여주는 cost map 

   



