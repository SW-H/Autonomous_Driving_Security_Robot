from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
from PIL import ImageGrab
import cv2
import socket
import pickle

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tool import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from collections import deque


def detect_cv2_camera():
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
    class_names = load_class_names(namesfile)
    class_names2 = load_class_names(namesfile2)
    pts = [deque(maxlen=30) for _ in range(9999)]
    max_cosine_distance = 0.2
    nn_budget = None
    nms_max_overlap = 1.0

    model_filename = 'data/market1501.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    frame_num = 0

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
        draw_boxes = []
        person_draw_boxes = []
        person_draw_boxes_show = []
        print_data = [start,frame_num,[],[],[],[],[]]
        for box in boxes[0]:
            x1 = box[0]*width
            y1 = box[1]*height
            x2 = box[2]*width
            y2 = box[3]*height
            draw_boxes.append([x1,y1,x2,y2])
            print_data[2].append([box[5],box[6]])
        print_data[3] = draw_boxes

        for person_box in person_boxes[0]:
            if person_box[6] == 0:
                x1 = person_box[0]*width
                y1 = person_box[1]*height
                x2 = person_box[2]*width
                y2 = person_box[3]*height
                person_draw_boxes.append([x1,y1,x2,y2])
                person_draw_boxes_show.append(person_box)
                print_data[4].append([person_box[5],person_box[6]])
        print_data[5] = person_draw_boxes
        features = encoder(img_showed,person_draw_boxes)

        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(person_draw_boxes, features)]        
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxs, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

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
            count += 1

        img_showed = cv2.resize(img_showed, (400,200))
        finish = time.time()
        print('Predicted in %f seconds.' % (finish - start))
        result_img = plot_boxes_cv2(img_showed, boxes[0], savename=None, class_names=class_names)
        result_img = plot_boxes_cv2(result_img, person_draw_boxes_show, savename=None, class_names=class_names2,color=(0,255,255))
        cv2.imshow('PHITITNAS panorama camera', result_img)
        frame_num += 1

        sock.send(str(print_data).encode())
        
        if cv2.waitKey(50) & 0xFF == ord('q'):
            sock.close()
            break

if __name__ == '__main__':
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('192.168.70.3',7008))
    detect_cv2_camera() 