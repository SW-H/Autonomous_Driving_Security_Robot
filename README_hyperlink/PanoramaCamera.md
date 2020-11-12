# Panorama Camera 

+ 단위시간 당 시각적인 데이터를 최대한 많이 수집하기 위해 자율주행 로봇에 360도 파노라마 카메라를 장착하였다.   
카메라의 자체 소프트웨어 버전을 확인하고 그에 맞는 툴을 제공받기 위해 제조사와 수차례 연락을 주고받았다.   카메라의 버전(2.3.1)에 맞추기 위해 제공받은 소프트웨어의 버전(3.5.5)을 조정하여 사용하였다.![PC에서 파노라마 카메라를 감지하여 연결하는 화면](https://github.com/SW-H/Autonomous_Driving_Security_Robot/blob/main/README_img/screenshot_of_software.PNG)   
↳PC에서 파노라마 카메라를 감지하여 연결하는 화면   


+ 카메라에 자체 내장된 무선AP와 PC를 연결하여 이미지를 전달받았고, PC는 유선LAN을 통해 네트워크에 접근하였다.   
![카메라와 연결에 성공하여 실시간으로 이미지를 전송받는 화면](https://github.com/SW-H/Autonomous_Driving_Security_Robot/blob/main/README_img/screenshot_of_transporting_img.PNG)
↳ 카메라와 연결에 성공하여 실시간으로 이미지를 전송받는 화면

+ RTMP방식으로 Live Streaming을 진행하여 실시간 촬영 이미지를 얻어올 경우, 그 연결이 매우 불안정하고 딜레이 시간이 너무 커 real-time service를 구현하기에 한계가 있다고 판단하였다.   또한 촬영이미지의 배포는 불필요하며, 관리자에게만 공개되야 할 필요가 있었다.   

   따라서 서버를 거치지 않고 OpenCV를 이용해 스크린을 실시간 캡쳐하는 방식으로 이미지를 수집하였다.   
카메라 툴 자체에서 저장되는 실시간 이미지는 3840 * 1920의 해상도를 갖고 있으나, 파이썬 모듈을 이용해 실제 유효한 영역만을 1145 * 466의 해상도로 스크린 캡쳐하여 사용했다.   

   이는 사람이 불편함을 느끼지 않는 해상도이며, AI model은 resize를 통해 축소된 이미지를 사용하므로 높은 해상도의 이미지가 불필요하다.   따라서, ImageGrab방식이 실제 서비스 제공에 문제되지 않는다고 판단하였다.   그 결과, AI model의 성능에 악영향을 주지 않으면서 이미지 수집에 걸리는 딜레이 시간을 획기적으로 줄일 수 있었다.   ![카메라에 증폭기를 장착한 모습](https://github.com/SW-H/Autonomous_Driving_Security_Robot/blob/main/README_img/amplifier_connected.PNG)   
↳카메라에 증폭기를 장착한 모습      


+ 또한 증폭기를 장착한 카메라를 드론과 연결하여 공중에서 높은 프레임과 화질로 Live Streaming 하는 사례를 확인하였다. 
이를 참고하여 무선AP의 유효 범위를 늘리고 통신 속도를 높이기 위해 추가로 증폭기를 장착하였다.  결과적으로, 로봇의 사용 범위가 크게 늘어나고 프레임 드랍 및 화질 저하가 줄어드는 성과를 거두었다.

