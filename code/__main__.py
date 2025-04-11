import sys
import _MyModel
from UI import QApplication, ChatGPTUI
import os
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "D:\不会编程\Machine_Learning\class_project\project\.venv\Lib\site-packages\PyQt5\Qt5\plugins"


if __name__ == '__main__':
    myModel = _MyModel.MyModel()
    app = QApplication(sys.argv)
    window = ChatGPTUI(myModel)
    window.show()
    window.remove_chat_item(window.first_list_item)
    window.create_new_chat()
    sys.exit(app.exec_())
