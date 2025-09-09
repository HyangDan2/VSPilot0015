import sys, asyncio
from PySide6.QtWidgets import QApplication
from qasync import QEventLoop
from ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    w = MainWindow()
    w.show()

    with loop:
        loop.run_forever()

if __name__ == "__main__":
    main()
