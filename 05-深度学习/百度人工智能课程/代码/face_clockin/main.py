"""
主程序, 处理前端请求
"""""
import base64
import os

from flask import Flask, render_template, request, jsonify
from flask_migrate import Migrate, MigrateCommand
from flask_script import Manager
from flask_sqlalchemy import SQLAlchemy

# 初始化应用
app = Flask(__name__)

app.config['DEBUG'] = True
app.config['SECRET_KEY'] = base64.b64encode(os.urandom(32)).decode()
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:123456@127.0.0.1/school'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

from baidu_ai_face import bf

# 定义数据表的结构
db = SQLAlchemy(app)

class Student(db.Model):
    """
    定义学生信息表格结构
    """
    __tablename__ = 'students'
    # 标识符
    id = db.Column(db.Integer, primary_key=True)
    # 学生姓名
    st_name = db.Column(db.String(128), nullable=False)
    # 学生班级
    st_class = db.Column(db.String(128), nullable=False)
    # 学号
    st_num = db.Column(db.String(128), unique=True, nullable=False)
    # 性别
    st_gender = db.Column(db.Enum('MAN', 'WOMAN'), default='MAN')

    def to_dict(self):
        """
        把学生对象转换成字典数据
        :return: 字典数据
        """
        return {
            '姓名': self.st_name,
            '班级': self.st_class,
            '学号': self.st_num,
            '性别': self.st_gender
        }

# 关联应用和数据库操作
Migrate(app, db)
manager = Manager(app)
manager.add_command('db', MigrateCommand)

# 实现视图函数
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')

    # POST
    # 获取学生信息
    st_name = request.form.get('stu_name')
    st_class = request.form.get('stu_class')
    st_num = request.form.get('stu_num')
    st_gender = request.form.get('gender')

    if all([st_name, st_class, st_num, st_gender]) == False:
        return render_template('register.html', msg='信息不全,请重新填写')

    stu = Student()
    stu.st_name = st_name
    stu.st_class = st_class
    stu.st_num = st_num
    stu.st_gender = st_gender

    try:
        db.session.add(stu)
        db.session.commit()
    except Exception as e:
        print('添加数据库失败, {}'.format(e))
        return render_template('register.html', msg='信息错误, 请重新填写')

    # 获取照片
    img = request.form.get('img_data')

    # 添加照片到人脸库中
    ret = bf.add_user(img.split(',')[1], '1', st_num)

    if ret != 0:
        db.session.delete(stu)
        db.session.commit()
        return render_template('register.html', msg='照片不合格, 请重新拍照')

    return render_template('index.html')

@app.route('/clockin', methods=['GET', 'POST'])
def clockin():
    if request.method == 'GET':
        return render_template('clockin.html')

    # POST
    img = request.json.get('img_data')
    result = bf.search_user(img.split(',')[1], '1')

    if result['error'] == 0:
        # 根据user_id查询学员信息
        stu = Student.query.filter(Student.st_num == result['user_id']).first()
        if stu:
            return jsonify(errno=0, errmsg='识别成功', dict=stu.to_dict())
        else:
            return jsonify(errno=-1, errmsg='找不到学员信息')


    return jsonify(errno=result['error'], errmsg=result['err_msg'])

if __name__ == '__main__':
    manager.run()