from GUI_class import MyWindow
from PyQt6.QtWidgets import *
import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec())
