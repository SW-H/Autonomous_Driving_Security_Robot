import os
import secrets
from flask import *
import threading
from socket import *
import cv2
from PIL import ImageGrab
import numpy as np
import math
import time

from flask import render_template, url_for, flash, redirect,request
from flask_tutorial import app, db, bcrypt
from flask_tutorial.forms import RegistrationForm, LoginForm, UpdateAccountForm
from flask_tutorial.models import User, Post
from flask_login import login_user, current_user, logout_user, login_required
from flask_tutorial.tools.model_utils import plot_boxes_cv2
from flask_tutorial.forms import RegistrationForm, LoginForm
from flask_tutorial.face_recognition_cli import *



# TCP_IP = '192.168.70.3'
# TCP_PORT = 7008
# BUFFER_SIZE = 20

# TCP_IP_LOC = '192.168.70.3'
# TCP_PORT_LOC = 7012

# UDP_IP = '192.168.70.5'
# UDP_PORT = 40000

# masks = []
# persons = []

# msg=''

# square=(348,194,1493,660)
# img_output=np.zeros(square)

# def masked(masks,person):
#     for mask in masks:
#         if mask[5] == 0 and person[0] <= mask[0] and person[1] <= mask[1] and person[2] >= mask[2] and person[3] >= mask[3]:
#             return True
#     return False

# def gen_bboxed_frames(square):
#     class_names = ['with_mask', 'without_mask', 'mask_weared_incorrect']
#     class_names2 = ['person']
#     while True:
#         global img_output
#         cap = np.array(ImageGrab.grab(square))
#         ori_img = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
#         res_img = plot_boxes_cv2(ori_img, masks, class_names=class_names, color=None)
#         res_img = plot_boxes_cv2(res_img, persons, class_names=class_names2, color=(0,255,255))
#         joined_img=np.vstack([ori_img,res_img])
#         joined_img=np.vstack([joined_img,img_output])        
#         _, buffer = cv2.imencode('.jpg', joined_img)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                 b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# def face_recog(image_to_check):
#     tolerance=0.35
#     known_names, known_face_encodings = scan_known_people('known_people_folder')
#     return test_image(image_to_check, known_names, known_face_encodings, tolerance)



@app.route('/')         #app.route는 기호를 통하여 다른 페이지 정의


@app.route('/home')
def home():
    return render_template('home.html')


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

@app.route('/login', methods=['GET','POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            flash('Login successful!' ,'success')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html', title='Login', form=form)

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))

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

@app.route("/controller")
@login_required
def controller():
    flash('12:07:47 GMT+0900 (대한민국 표준시) - connected!')
    return render_template('controller.html', title='Controller')

# @app.route("/about")
# def about():
#     return render_template('about.html')

# @app.route('/model_result')
# def model_result():
#     global detection_result
#     serverSocket = socket(AF_INET, SOCK_STREAM)
#     serverSocket.bind((TCP_IP,TCP_PORT))
#     serverSocket.listen(0)
#     connectionSocket, _ = serverSocket.accept()
   
#     while True:
#         data = connectionSocket.recv(8192)
#         detection_result = data.decode()
#         connectionSocket.send('success'.encode())
#         serverSocket.close()
#     return "model_result"

# @app.route('/ordering')
# def ordering():
#     sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
#     global img_output
#     global msg
#     state = 0
#     square=(348,194,1493,660)
#     distance = 1
#     while True:
#         global detection_result
#         global masks
#         global persons
#         data = eval(detection_result)
#         start = data[0]
#         frame_num = data[1]
#         mask_prob = data[2]
#         mask_bbox = data[3]
#         person_prob = data[4]
#         person_bbox = data[5]
#         track_id = data[6]
#         masks, persons = [], []
#         targets = []

#         #prob+bbox
#         for idx in range(len(mask_prob)):
#             masks.append(mask_bbox[idx] + mask_prob[idx]) #[x1,y1,x2,y2,prob,label]

#         for idx in range(len(person_prob)):
#             persons.append(person_bbox[idx] + person_prob[idx])
#             persons[idx].append(track_id[idx]) #[x1,y1,x2,y2,prob,label,id]
            
#         #target selection
#         if state == 0:
#             #nearest unmasked
#             nearest_unmasked = []
#             max_area = 0
#             for mask in masks:
#                 if (mask[2] - mask[0]) * (mask[3] - mask[1]) >= max_area and mask[5]!=0:
#                     nearest_unmasked = mask
#                     max_area = (mask[2] - mask[0]) * (mask[3] - mask[1])
#             #person match
#             target = []
#             target_id = -1
#             if nearest_unmasked != [] and persons != []:
#                 for person in persons:
#                     if person[0] <= nearest_unmasked[0] and person[1] <= nearest_unmasked[1] and person[2] >= nearest_unmasked[2] and person[3] >= nearest_unmasked[3] and person[6] not in targets:
#                         target = person
#                         target_id = person[6]
#                         targets.append(target_id)
#                         state = 1
#                         img1_grab = np.array(ImageGrab.grab(square))
#                         img1_cvt = cv2.cvtColor(img1_grab, cv2.COLOR_BGR2RGB)
#                         break

#         #person tracking
#         if state == 1:
#             found = False
#             unmasked = True
#             for person in persons:
#                 if person[6] == target_id: #find correct target_id
#                     theta = ((person[2]+person[0])/2) / (square[2]-square[0]) * math.pi / 180 - math.pi
#                     if person[3] <= (square[3]-square[1]) * 0.9: #not close enough
#                         global rob_loc
#                         rob_loc = [0,0,0]
#                         z = theta+rob_loc[2]
#                         x = rob_loc[0] + math.cos(z) * distance
#                         y = rob_loc[1] + math.cos(z) * distance
#                         sock.send(str([x,y,z]).encode(),(UDP_IP,UDP_PORT)) #robot location send
#                     else: #close enough
#                         sock.send(str([5]).encode()) #TTS(마스크를 써주세요)
#                         img2_grab = np.array(ImageGrab.grab(square))
#                         img2_cvt = cv2.cvtColor(img2_grab, cv2.COLOR_BGR2RGB)
#                         TTS_time = time.time()
#                         delay_time = 0
#                         while delay_time < 10: #wait for 10seconds
#                             delay_time = time.time() - TTS_time
                            
#                             #current data load
#                             global detection_result
#                             global masks
#                             global persons
#                             data2 = eval(detection_result)
#                             start = data2[0]
#                             frame_num = data2[1]
#                             mask_prob = data2[2]
#                             mask_bbox = data2[3]
#                             person_bbox = data2[5]
#                             track_id = data2[6]
#                             masks = []
#                             for idx in range(len(mask_prob)):
#                                 masks.append(mask_bbox[idx] + mask_prob[idx]) #[x1,y1,x2,y2,prob,label]
#                             for idx in range(len(person_bbox)):
#                                 if track_id[idx] == target_id:
#                                     check = person_bbox[idx] #person to check

#                             if masked(masks,check): #check if masked or not
#                                 state = 0
#                                 unmasked = False
#                                 sock.sendto(str([6]).encode(),(UDP_IP,UDP_PORT)) #TTS(감사합니다)
#                                 break
#                         if unmasked: #unmasked until the end
#                             state = 0
#                             img_name = time.strftime('%Y-%m-%d', time.localtime(start))
#                             img_output = img2_cvt
#                             img_output_body = img2_body
#                             cv2.imwrite('./face_tutorial/criminal/'+img_name+'.jpg',img_output)
#                             cv2.imwrite('./face_tutorial/criminal/'+img_name+'body.jpg',img_output_body)
#                             msg = face_recog('./face_tutorial/criminal/'+img_name+'body.jpg')
#                     found = True
#             if found == False: #missed target kl,.l;;
#                 state = 0
#                 img_name = time.strftime('%Y-%m-%d', time.localtime(start))
#                 img_output = img1_cvt
#                 img_output_body = img1_body
#                 cv2.imwrite('./face_tutorial/criminal/'+img_name+'.jpg',img_output)
#                 cv2.imwrite('./face_tutorial/criminal/'+img_name+'body.jpg',img_output_body)
#                 msg = face_recog('./face_tutorial/criminal/'+img_name+'body.jpg')

#     return 'ordering'

# @app.route('/_stuff', methods = ['GET'])
# def stuff():
#     global msg
#     return jsonify(result=msg)


# @app.route('/robot_location')
# def robot_location():
#     global rob_loc
#     serverSocket = socket(AF_INET, SOCK_STREAM)
#     serverSocket.bind((TCP_IP_LOC,TCP_PORT_LOC))
#     serverSocket.listen(0)
#     connectionSocket, _ = serverSocket.accept()
   
#     while True:
#         data = connectionSocket.recv(8192)
#         rob_loc = eval(data.decode())
#         print(rob_loc)
#         connectionSocket.send('success'.encode())
#         serverSocket.close()
#     return "rob_loc"

# @app.route('/video_feed')
# def video_feed():
#     redirect(url_for('ordering'))
#     square = (348,194,1493,660)
#     return Response(gen_bboxed_frames(square),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/TTS')
# def TTS():
#     TTS_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
#     TTS_socket.connect('192.168.70.5',41000)
#     while True:
#         TTS_socket.send(TTS_text.encode())
#     return 'TTS'

# if __name__ == '__main__':
#     app.run(debug=True, port=12000, host = '172.22.77.52')       # flask run 커맨드를 해주지 않고도 flask server를 실행
