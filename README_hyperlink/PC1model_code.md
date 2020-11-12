
로봇에 장착된 카메라를 통해 수집된 이미지에서 목표한 기획에 맞게끔 자율주행 로봇의 움직임을 결정할 데이터를 도출하기 위해 다음과 같은 인공지능 모델들을 사용하였다.    
  
+ Mask Detection (YOLO v4) – Custom Data      
   파노라마 카메라로 수집한 이미지에서 마스크를 쓴 사람과 안 쓴 사람, 잘못 쓴 사람의 얼굴을 detection해내기 위한 CNN모델이다.   Kaggle에서 제공하는 VOC format의 Mask Detection Dataset을 convert2Yolo 툴을 이용해  YOLO에 맞는 데이터 형식으로 변환 후, Google Colab Pro 환경에서 직접 모델을 train시켜 weights값을 생성하였다.   이미지에서 마스크를 쓴 얼굴(with_mask), 마스크를 쓰지 않은 얼굴( without_mask), 마스크를 제대로 쓰지 않은 얼굴(mask_weared_incorrect)을 찾아낸다.               
   ![model_training](/README_img/model_training.PNG "model_training")    ↳ Colab Pro에서 진행한 model training이 완료된 화면과 이에 사용한 parameter   
   ![코드 실행 시 마스크 착용 여부에 따라 구분된 모습](/README_img/detecting_mask_nomask.PNG "코드 실행 시 마스크 착용 여부에 따라 구분된 모습")   ↳코드 실행 시 마스크 착용 여부에 따라 구분된 모습




+ Person Detection (YOLO v4) – Coco Dataset   
    Mask detection model만으로는 사람의 뒷모습을 잡아내지 못하여 한번 포착한 마스크 미착용자를 지속적으로 tracking할수가 없다. 따라서 사진 촬영 각도에 상관없이 이미지에서 사람을 detection 해낼 필요가 있었다.   
	 Detection 성능의 향상을 위해 Mask detection과 별개의 모델을 사용하였으며, coco dataset으로 훈련된 모델에서 ‘person’  label만을 사용하였다. ![detection_result](/README_img/detection_result.PNG "Coco dataset을 이용해 train한 모델의 detection 결과 예시
")    ↳Coco dataset을 이용해 train한 모델의 detection 결과 예시





+ Object Tracking (Deep-SORT) – Pretrained Model   
   앞에서 detection한 person의  bounding box를 tracking하는 모델이다. 수집된 이미지에서 person마다 각각의  label(track id)을 붙이고 tracking하기 위해 사용한다.![ObjectTracking](/README_img/ObjectTracking.PNG "Real-time으로 person detection & tracking 하는 모델 출력 예시")   ↳ Real-time으로 person detection & tracking 하는 모델 출력 예시
   사용하는 자율주행 로봇 및 카메라의 특성을 고려하여, 연속적으로 촬영한 이미지에서의 원활한 tracking을 위해 model의 hyper parameter들을 조정하였다.   (max_iou_distance = 0.7, max_cos_distance = 0.2)




+ Face Recognition (dlib + face_recognition)   
   촬영된 이미지에서 Detection된 face를 database에 저장된 face들과 비교해서 개개인을 식별하고 등록되지 않은 face(unknown)를 색출하기 위해 사용한다. 계속해서 업데이트 되고있는 face_recognition api를 사용하며, 이는 전세계 사람들의 얼굴 데이터인 Labeled Faces in the Wild를 기준으로 99.38%의 정확도를 기록하였다.   ![dlib_and_face_recognition](/README_img/dlib_and_face_recognition.PNG "Face Recognition model 사용 예시")   ↳ Face Recognition model 사용 예시





### PC1 Model Code 
```python
m = Darknet('./cfg/yolov4.cfg')
m.print_network()
m.load_weights('./yolov4.weights')
print('Loading weights from %s... Done!' % ('./yolov4.weights'))
m.cuda()

m2 = Darknet('./cfg/person_yolov4.cfg')
m2.print_network()
m2.load_weights('./person_yolov4.weights')
print('Loading weights from %s... Done!' % ('./person_yolov4.weights'))
m2.cuda()

namesfile = 'data/obj.names'
namesfile2 = 'data/obj2.names'
```
 ↳ YOLO v4(object detection)에 적용할 각각의 weights, config, names.list파일 불러오기   

```python
model_filename = 'data/market1501.pb'
encoder = gdet.create_box_encoder(model_filename,batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)
```
 ↳ Deep_SORT(object tracking)에 적용할 모델 불러오기   

 ```python
 while True:
    square=(348,194,1493,660)
    img = np.array(ImageGrab.grab(bbox=square))
    img_model_input = cv2.resize(img,(m.width,m.height))
    img_model_input2 = cv2.resize(img,(m2.width,m2.height))
    img_showed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    width, height = (square[2]-square[0]),(square[3]-square[1])
    start = time.time()
    boxes = do_detect(m, img_model_input, 0.6, 0.6, True)
    person_boxes = do_detect(m2, img_model_input2, 0.6, 0.6, True)
 ```
 ↳ Real-time으로 파노라마 카메라 촬영 이미지 가져오기   
 
  ```python
         tracker.predict()
    tracker.update(detections)
 
    center_pt = []
    count = 0
 
    for track in tracker.tracks:
        print_data[6].append(track.track_id)
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        center = (int(((bbox[0])+(bbox[2]-bbox[0]))/2),int(((bbox[1])+(bbox[3]-bbox[1]))/2))
        center_pt.append(center)
            
        pts[track.track_id].append(center)
        thickness = 5  
        cv2.circle(img_showed,  (center), 1, (255,0,0), thickness)    
 
        for j in range(1, len(pts[track.track_id])):
            if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
               continue
            thickness = int(np.sqrt(64 / float(j + 1)) * 2)
            cv2.line(img_showed,(pts[track.track_id][j-1]), (pts[track.track_id][j]),(255,0,0),thickness)
   ``` 
   ↳ Detection결과에 대한 tracking   
   
   ```python
        print('Predicted in %f seconds.' % (finish - start))
        result_img = plot_boxes_cv2(img_showed, boxes[0], savename=None, class_names=class_names)
        result_img = plot_boxes_cv2(result_img, person_draw_boxes_show, savename=None, class_names=class_names2,color=(0,255,255))
        cv2.imshow('PHITITNAS panorama camera', result_img)
        frame_num += 1
   ```     
   ↳ 원본이미지에 bounding box 이미지를 덮어 출력   





   ```
def face_rocog(image_to_check):
    tolerance=0.35
    known_names, known_face_encodings = scan_known_people('known_people_folder')
    test_image(image_to_check, known_names, known_face_encodings, tolerance)

    
   ```
   ↳ target image를 사전에 저장된 사진들에 대해 face_recognition model을 실행시키는 부분






