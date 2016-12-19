import sys
from PyQt4 import QtGui, uic

from PyQt4.QtCore import QObject, pyqtSignal
import logging
import cedl
logger = logging.getLogger(cedl.__name__)
logging.basicConfig(level=logging.DEBUG)

import threading
 
class FuncThread(threading.Thread):
    def __init__(self, target, finalizer, *args):
        self._target = target
        self._args = args
        self._finalizer = finalizer
        threading.Thread.__init__(self)
 
    def run(self):
        self._target(*self._args)
        self._finalizer()
 
 


class QHandler(QObject, logging.Handler):

    messageReceived = pyqtSignal(str, int, str)

    def __init__(self, parent=None, level=logging.NOTSET):
        QObject.__init__(self, parent)
        logging.Handler.__init__(self, level)

    def emit(self, record):
        try:
            msg = self.format(record)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)
        self.messageReceived.emit(msg, record.levelno, record.name)

class MyWindow(QtGui.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        uic.loadUi('C:/Users/sommerc/Documents/cedlGUI/cedlguimain.ui', self)
        
        self.make_connections()
        self.setup_logging()
        
        logger.info("Starting mainwindow")
        
    def setup_logging(self):
        self.handler = QHandler(self)
        self.handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)-8s\t%(message)s')
        self.handler.setFormatter(formatter)
        self.handler.messageReceived.connect(self.showMessage)
        
        logger.addHandler(self.handler)
        
        textedit = self.log_widget
        textedit.setReadOnly(True)
        textedit.setMaximumBlockCount(0)
        textedit.setCenterOnScroll(True)
        format_ = QtGui.QTextCharFormat()
        format_.setFontFixedPitch(True)
        textedit.setCurrentCharFormat(format_)

        
    def make_connections(self):
        self.pb_ch5_input_train.clicked.connect(self.cb_ch5_input_train)
        self.pb_ch5_input_encode.clicked.connect(self.cb_ch5_input_encode)
        
        self.pb_plate_mapping_train.clicked.connect(self.cb_plate_mapping_train)
        self.pb_plate_mapping_encode.clicked.connect(self.cb_plate_mapping_encode)
        
        self.pb_autoencoder_model.clicked.connect(self.cb_autoencoder_model)
        
        
        self.pb_start_training.clicked.connect(self.start_training)
        self.pb_start_encode.clicked.connect(self.start_encode)
    
    def showMessage(self, msg, level=None, name=None):
        
        self.log_widget.appendHtml(msg.replace('\n', '<br>'))
        self.log_widget.moveCursor(QtGui.QTextCursor.End)    
        
    def cb_ch5_input_train(self):
        fn = ''
        fn = QtGui.QFileDialog.getOpenFileName(self, "Select a CellH5 file",
                        "~",
                        "CellH5 (*.ch5 *.c5 *.h5 *.hdf)")
        self.ch5_input_train.setText(fn)
        
    def cb_ch5_input_encode(self):
        fn = ''
        fn = QtGui.QFileDialog.getOpenFileName(self, "Select a CellH5 file",
                        "~",
                        "CellH5 (*.ch5 *.c5 *.h5 *.hdf)")
        self.ch5_input_encode.setText(fn)
        
    def cb_plate_mapping_train(self):
        fn = ''
        fn = QtGui.QFileDialog.getOpenFileName(self, "Select a plate mapping file",
                        "~",
                        "TXT (*.txt *.*)")
        self.plate_mapping_train.setText(fn)
        
    def cb_plate_mapping_encode(self):
        fn = ''
        fn = QtGui.QFileDialog.getOpenFileName(self, "Select a plate mapping file",
                        "~",
                        "TXT (*.txt *.*)")
        self.plate_mapping_encode.setText(fn)
        
    def cb_autoencoder_model(self):
        fn = ''
        fn = QtGui.QFileDialog.getExistingDirectory(self, "Select a plate mapping file")
        self.autoencoder_model.setText(fn)
        
        
        
    def start_training(self):
        targs = cedl.argparse.Namespace()
        
        targs.cellh5_input = str(self.ch5_input_train.text())
        targs.cellh5_mapping = str(self.plate_mapping_train.text())
        targs.im_size = self.bbox_size.value()
        
        targs.ae_arch = str(self.autoencoder_architecture.text())
        
        targs.learning_rate = self.learning_rate.value()
        targs.momentum = self.momentum.value()
        targs.batchsize = self.batchsize.value()
        targs.epochs = self.epochs.value()
        targs.nsamples = self.nsamples.value()
        targs.corruption = self.corruption.value()
        targs.learner = str(self.learner.currentText()).lower()
        
        targs.neg_condition = str(self.neg_indicator.text())
        
        self.pb_start_training.setEnabled(False)
        self.pb_start_encode.setEnabled(False)
        self.train_thread = FuncThread(cedl.train, self.finalize_train, targs)
        self.train_thread.start()

    def finalize_train(self):
        logger.info("\nTraining finished\n*********************\n")
        self.pb_start_training.setEnabled(True)
        self.pb_start_encode.setEnabled(True)
        
        self.ch5_input_encode.setText(self.ch5_input_train.text())
        self.plate_mapping_encode.setText(self.plate_mapping_train.text())
        self.autoencoder_model.setText(cedl.get_model_name(str(self.ch5_input_train.text()), str(self.autoencoder_architecture.text())))
        
    def finalize_encode(self):
        logger.info("\nFeature encoding finished\n*********************\n")
        self.pb_start_encode.setEnabled(True)
        self.pb_start_training.setEnabled(True)

    def start_encode(self):
        eargs = cedl.argparse.Namespace()
        
        eargs.name = str(self.autoencoder_model.text())
        eargs.cellh5_input = str(self.ch5_input_encode.text())
        eargs.cellh5_mapping = str(self.plate_mapping_encode.text())
        eargs.im_size = self.bbox_size.value()
        
        self.pb_start_encode.setEnabled(False)
        self.pb_start_training.setEnabled(False)
        self.train_thread = FuncThread(cedl.predict, self.finalize_encode, eargs)
        self.train_thread.start()
        
    
    def check_training_params(self, t_params):
        pass
    

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())