import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np


class TransferModel(object):

    def __init__(self):

        # 定义训练和测试图片的变化方法，标准化以及数据增强
        self.train_generator = ImageDataGenerator(rescale=1.0 / 255.0)
        self.test_generator = ImageDataGenerator(rescale=1.0 / 255.0)

        # 指定训练数据和测试数据的目录
        self.train_dir = "./data/train"
        self.test_dir = "./data/test"

        # 定义图片训练相关网络参数
        self.image_size = (224, 224)
        self.batch_size = 32

        # 定义迁移学习的基类模型
        # 不包含VGG当中3个全连接层的模型加载并且加载了参数
        # vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
        self.base_model = VGG16(weights='imagenet', include_top=False)

        self.label_dict = {
            '0': 'bus',
            '1': 'dinosaurs',
            '2': 'elephants',
            '3': 'flowers',
            '4': 'horse'
        }

    def get_local_data(self):
        """
        读取本地的图片数据以及类别
        :return: 训练数据和测试数据迭代器
        """
        # 使用flow_from_derectory
        train_gen = self.train_generator.flow_from_directory(self.train_dir,
                                                             target_size=self.image_size,
                                                             batch_size=self.batch_size,
                                                             class_mode='binary',
                                                             shuffle=True)
        test_gen = self.test_generator.flow_from_directory(self.test_dir,
                                                           target_size=self.image_size,
                                                           batch_size=self.batch_size,
                                                           class_mode='binary',
                                                           shuffle=True)
        return train_gen, test_gen

    def refine_base_model(self):
        """
        微调VGG结构，5blocks后面+全局平均池化（减少迁移学习的参数数量）+两个全连接层
        :return:
        """
        # 1、获取原notop模型得出
        # [?, ?, ?, 512]
        x = self.base_model.outputs[0]

        # 2、在输出后面增加我们结构
        # [?, ?, ?, 512]---->[?, 1 * 1 * 512]
        x = keras.layers.GlobalAveragePooling2D()(x)

        # 3、定义新的迁移模型
        x = keras.layers.Dense(1024, activation=tf.nn.relu)(x)
        y_predict = keras.layers.Dense(5, activation=tf.nn.softmax)(x)

        # model定义新模型
        # VGG 模型的输入， 输出：y_predict
        transfer_model = keras.models.Model(inputs=self.base_model.inputs, outputs=y_predict)

        return transfer_model

    def freeze_model(self):
        """
        冻结VGG模型（5blocks）
        冻结VGG的多少，根据你的数据量
        :return:
        """
        # self.base_model.layers 获取所有层，返回层的列表
        for layer in self.base_model.layers:
            layer.trainable = False

    def compile(self, model):
        """
        编译模型
        :return:
        """
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.sparse_categorical_crossentropy,
                      metrics=['accuracy'])
        return None

    def fit_generator(self, model, train_gen, test_gen):
        """
        训练模型，model.fit_generator()不是选择model.fit()
        :return:
        """
        # 每一次迭代准确率记录的h5文件
        modelckpt = keras.callbacks.ModelCheckpoint('./ckpt/transfer_{epoch:02d}-{val_acc:.2f}.h5',
                                                     monitor='val_acc',
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     mode='auto',
                                                     period=1)

        model.fit_generator(train_gen, epochs=3, validation_data=test_gen, callbacks=[modelckpt])

        return None

    def predict(self, model):
        """
        预测类别
        :return:
        """

        # 加载模型，transfer_model
        model.load_weights("./ckpt/transfer_02-0.93.h5")

        # 读取图片，处理
        image = load_img("./data/test/dinosaurs/400.jpg", target_size=(224, 224))
        image = img_to_array(image)
        print(image.shape)
        # 四维(224, 224, 3)---》（1， 224， 224， 3）
        img = image.reshape([1, image.shape[0], image.shape[1], image.shape[2]])
        print(img)
        # model.predict()

        # 预测结果进行处理
        image = preprocess_input(img)
        predictions = model.predict(image)
        print(predictions)
        res = np.argmax(predictions, axis=1)
        print(self.label_dict[str(res[0])])


if __name__ == '__main__':
    tm = TransferModel()
    # 训练
    # train_gen, test_gen = tm.get_local_data()
    # # print(train_gen)
    # # for data in train_gen:
    # #     print(data[0].shape, data[1].shape)
    # # print(tm.base_model.summary())
    # model = tm.refine_base_model()
    # # print(model)
    # tm.freeze_model()
    # tm.compile(model)
    #
    # tm.fit_generator(model, train_gen, test_gen)

    # 测试
    model = tm.refine_base_model()

    tm.predict(model)
