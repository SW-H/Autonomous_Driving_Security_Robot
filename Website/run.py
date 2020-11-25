from flask_tutorial import app

if __name__ == '__main__':
    app.run(debug=True, port=12000,host='172.30.1.3')       # flask run 커맨드를 해주지 않고도 flask server를 실행