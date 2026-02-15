import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# # 实现一个加法运算
# a = tf.constant(11.0)
# b = tf.constant(20.0)
#
# c = tf.add(a, b)
#
# # 获取默认图
# g = tf.get_default_graph()
# print("获取当前加法运算的图：", g)
#
# # 打印所有操作，张量默认图
# print(a.graph)
# print(b.graph)
# print(c.graph)
#
# # 2、创建另外一张图
# new_g = tf.Graph()
# with new_g.as_default():
#     new_a = tf.constant(11.0)
#     new_b = tf.constant(20.0)
#
#     new_c = tf.add(new_a, new_b)
#
# # 打印所有操作，张量默认图
# print(new_a.graph)
# print(new_b.graph)
# print(new_c.graph)
#
# # 指定一个会话运行tensorflow程序
# with tf.Session(graph=new_g) as sess:
#     print(sess.graph)
#     c_res = sess.run(new_c)
#     print(c_res)

# tensorboard
# # 实现一个加法运算
# a = tf.constant(11.0)
# b = tf.constant(20.0)
#
# c = tf.add(a, b)
#
# # 获取默认图
# g = tf.get_default_graph()
# print("获取当前加法运算的图：", g)
#
# # 打印所有操作，张量默认图
# print(a.graph)
# print(b.graph)
# print(c.graph)
#
# # 指定一个会话运行tensorflow程序
# with tf.Session() as sess:
#     print(sess.graph)
#
#     # 1、写入到events文件当中
#     filewriter = tf.summary.FileWriter("./tmp/summary/", graph=sess.graph)
#
#     c_res = sess.run(c)
#     print(c_res)

# # 实现一个加法运算
# con_a = tf.constant(3.0, name="con_a")
# con_b = tf.constant(4.0, name="con_b")
#
# sum_c = tf.add(con_a, con_b, name="sum_c")
#
#
# # 定义两个placeholder
# plt_a = tf.placeholder(dtype=tf.float32)
# plt_b = tf.placeholder(dtype=tf.float32)
#
# plt_add = tf.add(plt_a, plt_b)
#
# # print("打印con_a：\n", con_a)
# # print("打印con_b：\n", con_b)
# # print("打印sum_c：\n", sum_c)
#
#
# # 指定一个会话运行tensorflow程序
# # config=tf.ConfigProto(allow_soft_placement=True,
# #                                         log_device_placement=True)
# with tf.Session() as sess:
#     print(sess.graph)
#
#     # 1、写入到events文件当中
#     # filewriter = tf.summary.FileWriter("./tmp/summary/", graph=sess.graph)
#
#     # con_a, con_b, sum_c = sess.run([con_a, con_b, sum_c])
#     # print(con_a, con_b, sum_c)
#     # print(c_res)
#     res = sess.run(plt_add, feed_dict={plt_a: 5.0, plt_b: 6.0})
#     print(con_a.eval())
#     print(res)

# tensor
# con_1 = tf.constant(3.0)
# con_2 = tf.constant([1, 2, 3, 4])
# con_3 = tf.constant([[1, 2], [3, 4]])
# con_4 = tf.constant([ [[1, 2], [3, 4]], [[5, 2], [8, 4]]])
# print(con_1.shape, con_2.shape, con_3.shape, con_4.shape)
# 形状变化
# a_p = tf.placeholder(tf.float32, shape=[None, None])
# b_p = tf.placeholder(tf.float32, shape=[None, 10])
# c_p = tf.placeholder(tf.float32, shape=[2, 3])
#
# print("a_p shape:", a_p.get_shape())
# print("b_p shape:", b_p.get_shape())
# print("c_p shape:", c_p.get_shape())
#
# # 静态形状
# a_p.set_shape([5, 6])
# print("a_p shape:", a_p.get_shape())
# # # 1、张量形状本身已经固定的不能再次修改
# # a_p.set_shape([4, 3])
# # # 2、张量形状不能夸阶数修改
# # a_p.set_shape([30])
# # print("a_p shape:", a_p.get_shape())
#
# # 动态性状
# c_reshape_p = tf.reshape(c_p, [3, 2])
# print("c_p shape:", c_p.get_shape())
# print("c_reshape_p shape:", c_reshape_p.get_shape())

# 变量operation
# a = tf.Variable(initial_value=30.0)
# b = tf.Variable(initial_value=40.0)
#
# sum = tf.add(a, b)
#
# init_op = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     # 手动运行init_op
#     sess.run(init_op)
#     print("a", a)
#     print(sess.run(sum))

# 命令行参数
tf.app.flags.DEFINE_integer("max_step", 1000, "train step number")

FLAGS = tf.app.flags.FLAGS


def linearregression():
    """
    tensorflo实现线性回归
    :return:
    """
    # - 1
    # 准备好数据集：y = 0.8
    # x + 0.7
    # 100
    # x [100, 1]   0.8  + 0.7 = [100, 1]
    with tf.variable_scope("original_data"):
        X = tf.random_normal([100, 1], mean=0.0, stddev=1.0, name="original_data_x")
        # 0.8必须是个二维的形状
        y_true = tf.matmul(X, [[0.8]]) + [[0.7]]

    # 个样本
    # - 2
    # 建立线性模型
    # - 随机初始化W1和b1
    # 建立线性回归模型
    # w [1, 1]   b 1
    # [100, 1] x [1, 1] + [1] = [100, 1]
    # 初始化w, b必须使用变量op去初始化，需要被训练，
    with tf.variable_scope("linear_model"):
        weights = tf.Variable(initial_value=tf.random_normal([1, 1]), trainable=False, name="w")
        bias = tf.Variable(initial_value=tf.random_normal([1, 1]), name="b")

        y_predict = tf.matmul(X, weights) + bias

    # - y = W·X + b，目标：求出权重W和偏置b
    # - 3
    # 确定损失函数（预测值与真实值之间的误差）-均方误差
    # 均方误差 ((y - y_repdict)^2 ) / m = 得到平均每一个样本的误差
    # [100, 102, 99]   [23, 25, 21]
    with tf.variable_scope("loss"):
        loss = tf.reduce_mean(tf.square(y_predict - y_true))

    # - 4
    # 梯度下降优化损失：需要指定学习率（超参数）
    with tf.variable_scope("optimizer"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)

    # 1、收集观察张量
    tf.summary.scalar('losses', loss)
    tf.summary.histogram('weight', weights)
    tf.summary.histogram('biases', bias)

    # 2、合并收集的张量
    merge = tf.summary.merge_all()

    # 初始化变量op
    init_op = tf.global_variables_initializer()

    # 创建一个saver
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(init_op)

        filewriter = tf.summary.FileWriter("./tmp/summary/", graph=sess.graph)

        # print(weights.eval(), bias.eval())
        # saver.restore(sess, "./tmp/ckpt/linearregression")
        # print(weights.eval(), bias.eval())

        for i in range(FLAGS.max_step):
            sess.run(optimizer)
            # print("loss:", sess.run(loss))
            # print("weights:", sess.run(weights))
            # print("bias:", sess.run(bias))
            summary = sess.run(merge)

            filewriter.add_summary(summary, i)
            print("train loss:%f, weights:%f, bias:%f" % (loss.eval(), weights.eval(), bias.eval()))

            # checkpoint：检查点文件格式
            # tf.keras: h5
            saver.save(sess, "./tmp/ckpt/linearregression")

    return None


if __name__ == '__main__':
    linearregression()