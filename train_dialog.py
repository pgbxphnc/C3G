# Form implementation generated from reading ui file 'train_dialog.ui'
#
# Created by: PyQt6 UI code generator 6.8.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_Dialog_model_train(object):
    def setupUi(self, Dialog_model_train):
        Dialog_model_train.setObjectName("Dialog_model_train")
        Dialog_model_train.resize(400, 300)
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        Dialog_model_train.setFont(font)
        self.label = QtWidgets.QLabel(parent=Dialog_model_train)
        self.label.setGeometry(QtCore.QRect(30, 80, 351, 161))
        self.label.setWordWrap(True)
        self.label.setObjectName("label")

        self.retranslateUi(Dialog_model_train)
        QtCore.QMetaObject.connectSlotsByName(Dialog_model_train)

    def retranslateUi(self, Dialog_model_train):
        _translate = QtCore.QCoreApplication.translate
        Dialog_model_train.setWindowTitle(_translate("Dialog_model_train", "Dialog"))
        self.label.setText(_translate("Dialog_model_train", "We\'re sorry, but we are currently busy developing this feature. If you have any related needs, please return and click the link on the homepage to visit our GitHub."))
