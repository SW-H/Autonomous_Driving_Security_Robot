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

