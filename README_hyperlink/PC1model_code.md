
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






