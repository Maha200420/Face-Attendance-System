import sqlite3
import cv2
import os
from flask import Flask,request,render_template,redirect,session,url_for
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import time
from ftplib import FTP
import re



app = Flask(__name__)

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    if img != []:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

# def drivehqf():
#     drive = Drive()
#     drive.login("truprojects01", "Projects123@#")

#     # Upload a file
#     drive.upload(local_path=f"Attendance/Attendance-{datetoday}.csv", remote_path=f"/Attendance/Attendance-{datetoday}.csv")

#     # List files in a folder
#     files = drive.list_files(folder="/remote_folder")
#     for file in files:
#         print(file['filename'])

#     # Logout
#     drive.logout()


def drivehqf():
    try:
        # Connect to DriveHQ FTP server
        ftp = FTP('ftp.drivehq.com')
        ftp.login("truprojects01", "Projects123@#")
        local_file_path = f"Attendance/Attendance-{datetoday}.csv"
        remote_file_path=f"Attendance-{datetoday}.csv"
        # Change directory to the target directory on DriveHQ
        ftp.cwd('/')

        # Open the local file
        with open(local_file_path, 'rb') as file:
            # Upload the file to DriveHQ
            ftp.storbinary(f'STOR {remote_file_path}', file)
            print(f"File {local_file_path} uploaded successfully to {remote_file_path} on DriveHQ")
        
        # Close the FTP connection
        ftp.quit()
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if str(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')
        print(f"Attendance added for {username} - {userid} at {current_time}")
        drivehqf()
    else:
        print(f"{username} - {userid} already marked attendance for the day, but still, I am marking it")


@app.route('/index')
def index():
    names, rolls, times, l = extract_attendance()
    return render_template('index.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2, mess='Default message')

import base64

def decode_base64_image(base64_str):
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@app.route('/start', methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        names, rolls, times, l = extract_attendance()
        MESSAGE = 'This face is not registered with us, kindly register yourself first'
        return render_template('index.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                               datetoday2=datetoday2, mess=MESSAGE)
    
    names, rolls, times, l = extract_attendance()
    return render_template('index.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2, start_camera=True)

@app.route('/add', methods=['POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
        
    names, rolls, times, l = extract_attendance()
    return render_template('index.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2, register_camera=True, newusername=newusername, newuserid=newuserid)

@app.route('/api/verify_frame', methods=['POST'])
def api_verify_frame():
    try:
        data = request.get_json()
        image_data = data.get('image')
        img = decode_base64_image(image_data)
        if img is None:
            return {'status': 'error', 'message': 'Invalid image data'}
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        if len(faces) == 0:
            return {'status': 'no_face'}
            
        # Take the largest face found
        (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        face = cv2.resize(img[y:y+h, x:x+w], (50, 50))
        
        if 'face_recognition_model.pkl' not in os.listdir('static'):
            return {'status': 'no_model'}
            
        identified_person = identify_face(face.reshape(1, -1))[0]
        add_attendance(identified_person)
        
        username = identified_person.split('_')[0]
        userid = identified_person.split('_')[1]
        return {
            'status': 'success', 
            'identified_person': identified_person,
            'username': username,
            'userid': userid,
            'time': datetime.now().strftime("%H:%M:%S")
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@app.route('/api/register_frame', methods=['POST'])
def api_register_frame():
    try:
        data = request.get_json()
        image_data = data.get('image')
        newusername = data.get('newusername')
        newuserid = data.get('newuserid')
        frame_index = int(data.get('frame_index'))
        
        img = decode_base64_image(image_data)
        if img is None:
            return {'status': 'error', 'message': 'Invalid image data'}
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return {'status': 'no_face'}
            
        (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)
            
        name = f"{newusername}_{frame_index}.jpg"
        cv2.imwrite(os.path.join(userimagefolder, name), img[y:y+h, x:x+w])
        
        if frame_index == 49:
            print('Training Model on 50th frame')
            train_model()
            
        return {'status': 'success', 'frame_index': frame_index}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}
    


@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html")
    else:
        username = request.form.get('user','')
        name = request.form.get('name','')
        email = request.form.get('email','')
        number = request.form.get('mobile','')
        password = request.form.get('password','')

        # Server-side validation
        username_pattern = r'^.{6,}$'
        name_pattern = r'^[A-Za-z ]{3,}$'
        email_pattern = r'^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$'
        mobile_pattern = r'^[6-9][0-9]{9}$'
        password_pattern = r'^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$'

        if not re.match(username_pattern, username):
            return render_template("signup.html", message="Username must be at least 6 characters.")
        if not re.match(name_pattern, name):
            return render_template("signup.html", message="Full Name must be at least 3 letters, only letters and spaces allowed.")
        if not re.match(email_pattern, email):
            return render_template("signup.html", message="Enter a valid email address.")
        if not re.match(mobile_pattern, number):
            return render_template("signup.html", message="Mobile must start with 6-9 and be 10 digits.")
        if not re.match(password_pattern, password):
            return render_template("signup.html", message="Password must be at least 8 characters, with an uppercase letter, a number, and a lowercase letter.")

        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("SELECT 1 FROM info WHERE user = ?", (username,))
        if cur.fetchone():
            con.close()
            return render_template("signup.html", message="Username already exists. Please choose another.")
        
        cur.execute("insert into `info` (`user`,`name`, `email`,`mobile`,`password`) VALUES (?, ?, ?, ?, ?)",(username,name,email,number,password))
        con.commit()
        con.close()
        return redirect(url_for('login'))


@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "GET":
        return render_template("signin.html")
    else:
        mail1 = request.form.get('user','')
        password1 = request.form.get('password','')
        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
        data = cur.fetchone()
        if data == None:
            return render_template("signin.html", message="Invalid username or password.")    
        elif mail1 == 'admin' and password1 == 'admin':
            return render_template("index.html")
        elif mail1 == str(data[0]) and password1 == str(data[1]):
            return render_template("index.html")
        else:
            return render_template("signin.html", message="Invalid username or password.")

@app.route('/')
def home():
	return render_template('home.html')


@app.route('/notebook')
def notebook():
	return render_template('NOtebook.html')

if __name__ == '__main__':
    app.run(debug=True,port=5000)
