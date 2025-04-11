import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLineEdit, QListWidget, QLabel, QHBoxLayout, QListWidgetItem
import _MyModel
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtCore import Qt
class ChatSession:
    """储存单个对话的内容"""
    def __init__(self, topic="新对话"):
        self.topic = topic
        self.messages = []  # 存储聊天记录

    def add_message(self, sender, text):
        """添加消息（sender: 'user' 或 'ai'）"""
        self.messages.append((sender, text))


class ChatGPTUI(QWidget):

    def __init__(self, MyModel):
        super().__init__()
        self.model = MyModel
        self.first_list_item = QListWidget()
        self.setWindowTitle("ChatGPT 聊天界面")
        self.setGeometry(200, 200, 800, 600)
        self.setStyleSheet("background-color: #DCB272; color: white;")  # 设置深色背景
        #self.setWindowFlags(Qt.FramelessWindowHint) # 设置无边框
        # 创建主布局
        main_layout = QHBoxLayout(self)
        # 左侧：
        left_layout = QVBoxLayout()
        # 添加“新建对话”按钮
        self.new_chat_button = QPushButton("新建对话")
        self.new_chat_button.setStyleSheet("background-color: #0FA958; color: white; padding: 8px; border-radius: 5px;")
        self.new_chat_button.clicked.connect(self.create_new_chat)
        left_layout.addWidget(self.new_chat_button)
        # 左侧：对话历史列表
        self.history_list = QListWidget()
        self.history_list.setStyleSheet("background-color: #E4DECE; color: black; border: none;")
        self.history_list.itemClicked.connect(self.load_selected_chat)  # 绑定选择对话事件
        left_layout.addWidget(self.history_list)

        # 右侧：聊天区域
        right_layout = QVBoxLayout()

        # 对话主题输入框
        self.topic_input = QLineEdit()
        self.topic_input.setPlaceholderText("请输入对话主题...")
        self.topic_input.setStyleSheet("background-color: #E4DECE; color: black; padding: 5px; border-radius: 5px;")



        # 聊天显示区域
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("background-color: #E4DECE; color: black; border: none; padding: 10px;")
        right_layout.addWidget(self.chat_display, 7)

        # 输入区域（水平布局）
        input_layout = QHBoxLayout()

        # 用户输入框
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("输入消息...")
        self.input_field.setStyleSheet("background-color: #E4DECE; color: black; padding: 5px; border-radius: 5px;")
        input_layout.addWidget(self.input_field, 8)
        self.input_field.returnPressed.connect(self.send_message)

        # 发送按钮
        self.send_button = QPushButton("发送")
        self.send_button.setStyleSheet("background-color: #DA8D6D; color: white; padding: 8px; border-radius: 5px;")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button, 2)

        right_layout.addLayout(input_layout)

        # 将右侧布局添加到主布局
        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 8)

        self.setLayout(main_layout)

        # 初始对话存储
        self.chat_sessions = []  # 存储多个会话
        self.current_session = None
        self.create_new_chat()  # 启动时创建默认对话

    def create_new_chat(self):
        """新建对话并添加到历史列表"""
        topic = self.topic_input.text().strip()
        if not topic:
            topic = "新对话"

        new_session = ChatSession(topic)
        self.chat_sessions.append(new_session)
        self.current_session = new_session

        # 更新左侧历史对话列表
        self.add_chat_item(topic)
        self.history_list.setCurrentRow(self.history_list.count() - 1)  # 选中新建的对话
        self.chat_display.clear()

    def load_selected_chat(self):
        """切换到用户选择的历史对话"""
        selected_index = self.history_list.currentRow()
        if selected_index >= 0:
            self.current_session = self.chat_sessions[selected_index]
            self.display_chat_history()

    def display_chat_history(self):
        """显示当前会话的聊天记录"""
        self.chat_display.clear()
        for sender, text in self.current_session.messages:
            if sender == 'user':
                self.chat_display.append(f"<b><span style='color: #9b7438; font-family: 微软雅黑; font-size: 28px'>主题 : </span><span style='color: #1B2131; font-family: 微软雅黑; font-size: 28px'> {text}</span></b>")
            else:
                self.chat_display.append(f"<b>{'用户' if sender == 'user' else 'ChatGPT'}:</b> {text}")

    def send_message(self):
        """发送用户输入的消息"""
        user_text = self.input_field.text().strip()
        if user_text and self.current_session:
            self.current_session.add_message("user", user_text)
            self.chat_display.append(f"<b><span style='color: #9b7438; font-family: 微软雅黑; font-size: 28px'>主题 : </span><span style='color: #1B2131; font-family: 微软雅黑; font-size: 28px'> {user_text}</span></b>")
            self.input_field.clear()

            # 触发 AI 回复（暂时用占位内容）
            ai_reply = self.get_ai_response(user_text)
            self.receive_message(ai_reply)

    def receive_message(self, text):
        """显示 AI 回复"""
        if self.current_session:
            self.current_session.add_message("ai", text)
            self.chat_display.append(f"<b>ChatGPT:</b> {text}")

    def get_ai_response(self, user_input):
        """可在此接入 AI 模型，如 OpenAI API 或本地大模型"""
        output = self.model.predict(user_input)
        return f"<span style='font-size: 20px;'>{output}</span>"

    def add_chat_item(self, text):
        """ 添加带删除按钮的聊天记录项 """
        item_widget = QWidget()
        item_layout = QHBoxLayout(item_widget)
        item_layout.setContentsMargins(5, 2, 5, 2)

        label = QLabel(text)
        delete_button = QPushButton("×")
        delete_button.setFixedSize(20, 20)
        delete_button.setStyleSheet("background-color: #cc6666; color: white; border-radius: 10px;")

        item_layout.addWidget(label)
        item_layout.addWidget(delete_button)
        item_layout.addStretch()

        list_item = QListWidgetItem(self.history_list)
        list_item.setSizeHint(item_widget.sizeHint())

        self.history_list.addItem(list_item)
        self.history_list.setItemWidget(list_item, item_widget)

        # 绑定删除事件
        delete_button.clicked.connect(lambda: self.remove_chat_item(list_item))
        self.first_list_item = list_item

    def remove_chat_item(self, item):
        """ 删除聊天记录项 """
        row = self.history_list.row(item)
        del self.chat_sessions[row]
        self.history_list.takeItem(row)


# 运行 PyQt5 应用
if __name__ == "__main__":

    app = QApplication(sys.argv)
    window = ChatGPTUI()
    window.show()
    window.remove_chat_item(window.first_list_item)
    window.create_new_chat()
    sys.exit(app.exec_())
