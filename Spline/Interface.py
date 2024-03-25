import ctypes
import sys
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt

Num = 10000
lib = ctypes.CDLL('./lib/bin/spline.so')
lib.get_approximation.restype = ctypes.POINTER(ctypes.c_double)
lib.get_approximation.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_uint32]
lib.get_error.restype = ctypes.POINTER(ctypes.c_double)
lib.get_error.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_uint32]
lib.get_f.restype = ctypes.POINTER(ctypes.c_double)
lib.get_f.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_uint32]
lib.get_f1.restype = ctypes.POINTER(ctypes.c_double)
lib.get_f1.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_uint32]
lib.get_f2.restype = ctypes.POINTER(ctypes.c_double)
lib.get_f2.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_uint32]
lib.get_s1.restype = ctypes.POINTER(ctypes.c_double)
lib.get_s1.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_uint32]
lib.get_s2.restype = ctypes.POINTER(ctypes.c_double)
lib.get_s2.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_uint32]
lib.get_derivative_error.restype = ctypes.POINTER(ctypes.c_double)
lib.get_derivative_error.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_uint32]
lib.get_second_derivative_error.restype = ctypes.POINTER(ctypes.c_double)
lib.get_second_derivative_error.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_uint32]
lib.get_a.restype = ctypes.POINTER(ctypes.c_double)
lib.get_a.argtypes = [ctypes.c_uint32]
lib.get_b.restype = ctypes.POINTER(ctypes.c_double)
lib.get_b.argtypes = [ctypes.c_uint32]
lib.get_c.restype = ctypes.POINTER(ctypes.c_double)
lib.get_c.argtypes = [ctypes.c_uint32]
lib.get_d.restype = ctypes.POINTER(ctypes.c_double)
lib.get_d.argtypes = [ctypes.c_uint32]
lib.set_spline.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_uint32, ctypes.c_int, ctypes.c_double, ctypes.c_double]
lib.free_some.argtypes = [ctypes.c_int]


def setSpline(a: float, b: float, n: int, fun_num: int, mu1: float = 0, mu2: float = 0):
    lib.set_spline(a, b, n, fun_num, mu1, mu2)

def getApproximation(a: float, b: float, n: int):
    ptr = lib.get_approximation(a, b, n)
    return ptr[:n]

def getA( n: int):
    ptr = lib.get_a(n)
    return ptr[:n]

def getB( n: int):
    ptr = lib.get_b(n)
    return ptr[:n]

def getC(n: int):
    ptr = lib.get_c(n)
    return ptr[:n]

def getD( n: int):
    ptr = lib.get_d(n)
    return ptr[:n]

def getError(a: float, b: float, n: int):
    ptr = lib.get_error(a, b, n)
    return ptr[:n ]

def getDerivativeError(a: float, b: float, n: int):
    ptr = lib.get_derivative_error(a, b, n)
    return ptr[:n ]

def getSecondDerivativeError(a: float, b: float, n: int):
    ptr = lib.get_second_derivative_error(a, b, n)
    return ptr[:n ]

def FreeAll():
    lib.free_all()

def FreeSome(i: int):
    lib.free_some(i)

def getF(a: float, b: float, n: int):
    ptr = lib.get_f(a, b, n)
    return ptr[:n ]

def getF1(a: float, b: float, n: int):
    ptr = lib.get_f1(a, b, n)
    return ptr[:n ]

def getF2(a: float, b: float, n: int):
    ptr = lib.get_f2(a, b, n)
    return ptr[:n ]

def getS1(a: float, b: float, n: int):
    ptr = lib.get_s1(a, b, n)
    return ptr[:n ]

def getS2(a: float, b: float, n: int):
    ptr = lib.get_s2(a, b, n)
    return ptr[:n ]

class Window(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Сплайны")
        self.setGeometry(100, 100, 1850, 1000)

        #making graph template
        self.plot_widget = pg.PlotWidget(self)
        self.plot_widget.setGeometry(460, 10, 1000, 990)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setBackground('w')
        self.plot_widget.show()

        #making tables
        self.table1 = QTableWidget(self)
        self.table1.setGeometry(0, 350, 450, 650)
        self.table1.setColumnCount(10)
        self.table1.setHorizontalHeaderLabels(["x", "F(x)", "S(x)","a", "b", "c", "d", "|f(x) - S(x)|", "|f'(x) - S'(x)|", "|f''(x) - S''(x)|"])

        self.table2 = QTableWidget(self)
        self.table2.setGeometry(1480, 10, 350, 940)
        self.table2.setColumnCount(3)
        self.table2.setHorizontalHeaderLabels(["x", "F(x)", "error(x)"])

        #list of functions
        text1 = QtWidgets.QLabel(self)
        text1.setText("Функция")
        text1.setGeometry(85, 0, 100, 40)
        self.combo_box = QComboBox(self)
        self.combo_box.addItem("Тестовая")
        self.combo_box.addItem("F(x) = sin(exp(x))")
        self.combo_box.addItem("F(x) = sin(cos(x))")
        self.combo_box.addItem("F(x) = sin(x) / x")
        self.combo_box.addItem("F(x) = exp(x - 3)")
        self.combo_box.setGeometry(10, 40, 210, 40)

        #S"(a) = mu1; S"(b) = mu2
        text4 = QtWidgets.QLabel(self)
        text4.setText("Граничные условия")
        text4.setGeometry(250, 15, 140, 25)
        self.line_edit1 = QLineEdit(self)
        self.line_edit1.setGeometry(300, 45, 90, 30)
        self.line_edit1.setText('0')
        self.line_edit2 = QLineEdit(self)
        self.line_edit2.setGeometry(300, 85, 90, 30)
        self.line_edit2.setText('0')
        text2 = QtWidgets.QLabel(self)
        text2.setText("S''(a) = ")
        text2.setGeometry(240, 45, 70, 30)
        text3 = QtWidgets.QLabel(self)
        text3.setText("S''(b) = ")
        text3.setGeometry(240, 85, 70, 30)

        #n
        text5 = QtWidgets.QLabel(self)
        text5.setText("Число разбиений")
        text5.setGeometry(10, 90, 120, 30)
        self.line_edit3 = QLineEdit(self)
        self.line_edit3.setGeometry(130, 90, 90, 30)
        self.line_edit3.setText('10')

        #button approximation
        self.button = QtWidgets.QPushButton(self)
        self.button.setGeometry(240, 125, 200, 30)
        self.button.setText("Аппроксимировать")
        self.button.clicked.connect(self.Click)

        #button clear graph
        self.button1 = QtWidgets.QPushButton(self)
        self.button1.setGeometry(1500, 960, 150, 30)
        self.button1.setText("Очистить график")
        self.button1.clicked.connect(self.Clear)

        #a, b
        text6 = QtWidgets.QLabel(self)
        text6.setText("a = ")
        text6.setGeometry(10, 130, 30, 30)
        self.line_edit4 = QLineEdit(self)
        self.line_edit4.setGeometry(40, 130, 60, 30)
        self.line_edit4.setText('-1')
        text7 = QtWidgets.QLabel(self)
        text7.setText("b = ")
        text7.setGeometry(110, 130, 30, 30)
        self.line_edit5 = QLineEdit(self)
        self.line_edit5.setGeometry(140, 130, 60, 30)
        self.line_edit5.setText('1')

        #list of functions
        text8 = QtWidgets.QLabel(self)
        text8.setText("График")
        text8.setGeometry(45, 160, 100, 30)
        self.combo_box2 = QComboBox(self)
        self.combo_box2.addItem("F(x), S(x)")
        self.combo_box2.addItem("F'(x), S'(x)")
        self.combo_box2.addItem("F''(x), S''(x)")
        self.combo_box2.setGeometry(10, 190, 120, 30)

        #maxs
        text9 = QtWidgets.QLabel(self)
        text9.setText("max |F(x) - S(x)| = ")
        text9.setGeometry(10, 230, 120, 30)
        self.text9_1 = QtWidgets.QLabel(self)
        self.text9_1.setStyleSheet('background-color: white;')
        self.text9_1.setGeometry(130, 230, 120, 30)
        text9_2 = QtWidgets.QLabel(self)
        text9_2.setText(" в x =")
        text9_2.setGeometry(260, 230, 40, 30)
        self.text9_3 = QtWidgets.QLabel(self)
        self.text9_3.setStyleSheet('background-color: white;')
        self.text9_3.setGeometry(310, 230, 100, 30)

        text10 = QtWidgets.QLabel(self)
        text10.setText("max |F'(x) - S'(x)| = ")
        text10.setGeometry(10, 270, 120, 30)
        self.text10_1 = QtWidgets.QLabel(self)
        self.text10_1.setStyleSheet('background-color: white;')
        self.text10_1.setGeometry(135, 270, 120, 30)
        text10_2 = QtWidgets.QLabel(self)
        text10_2.setText(" в x =")
        text10_2.setGeometry(265, 270, 40, 30)
        self.text10_3 = QtWidgets.QLabel(self)
        self.text10_3.setStyleSheet('background-color: white;')
        self.text10_3.setGeometry(315, 270, 100, 30)

        text11 = QtWidgets.QLabel(self)
        text11.setText("max |F''(x) - S''(x)| = ")
        text11.setGeometry(10, 310, 130, 30)
        self.text11_1 = QtWidgets.QLabel(self)
        self.text11_1.setStyleSheet('background-color: white;')
        self.text11_1.setGeometry(140, 310, 120, 30)
        text11_2 = QtWidgets.QLabel(self)
        text11_2.setText(" в x =")
        text11_2.setGeometry(270, 310, 40, 30)
        self.text11_3 = QtWidgets.QLabel(self)
        self.text11_3.setStyleSheet('background-color: white;')
        self.text11_3.setGeometry(320, 310, 100, 30)
    
    def Clear(self):
        self.plot_widget.clear()

    def Click(self):
        mu1 = (float)(self.line_edit1.text())
        mu2 = (float)(self.line_edit2.text())
        n = (int)(self.line_edit3.text())
        fun_num = self.combo_box.currentIndex()
        a = (float)(self.line_edit4.text())
        b = (float)(self.line_edit5.text())
        N = Num*(int)(abs(b-a))
        setSpline(a, b, n, fun_num, mu1, mu2)
        x_vals = np.linspace(a, b, N)
        if (self.combo_box2.currentIndex() == 0):
            s = getApproximation(a, b, N)
            f = getF(a, b, N)
            error = getError(a, b, N)
        elif (self.combo_box2.currentIndex() == 1):
            s = getS1(a, b, N)
            f = getF1(a, b, N)
            error = getDerivativeError(a, b, N)
        else :
            s = getS2(a, b, N)
            f = getF2(a, b, N)
            error = getSecondDerivativeError(a, b, N)
        self.plot_widget.plot(x_vals, error, pen = 'r')
        self.plot_widget.plot(x_vals, s, pen = 'b')
        self.plot_widget.plot(x_vals, f, pen = 'g')
        self.plot_widget.show()

        self.table2.clearContents()
        self.table2.setRowCount(0)
        if (self.combo_box2.currentIndex() == 0):
            self.table2.setHorizontalHeaderLabels(["x", "F(x)", "S(x)"])
        elif (self.combo_box2.currentIndex() == 1):
            self.table2.setHorizontalHeaderLabels(["x", "F'(x)", "S'(x)"])
        else:
            self.table2.setHorizontalHeaderLabels(["x", "F''(x)", "S''(x)"])

        for i in range(N):
            self.table2.insertRow(i)
            itemx = QTableWidgetItem()
            itemx.setData(Qt.DisplayRole, "{:.9f}".format(x_vals[i]) )
            self.table2.setItem(i, 0, itemx)
            itemf = QTableWidgetItem()
            itemf.setData(Qt.DisplayRole, "{:.9f}".format(f[i]) )
            self.table2.setItem(i, 1, itemf)
            items = QTableWidgetItem()
            items.setData(Qt.DisplayRole, "{:.9f}".format(s[i]) )
            self.table2.setItem(i, 2, items)

        FreeSome(self.combo_box2.currentIndex())

        self.table1.clearContents()
        self.table1.setRowCount(0)
        s = getApproximation(a, b, n)
        x_vals = np.linspace(a, b, n)
        f = getF(a, b, n)
        error = getError(a, b, n)
        error1 = getDerivativeError(a, b, n)
        error2 = getSecondDerivativeError(a, b, n)
        max_ind0 = error.index(max(error))
        max_ind1 = error1.index(max(error1))
        max_ind2 = error2.index(max(error2))
        self.text9_1.setText("{:.3e}".format(error[max_ind0]))
        self.text9_3.setText("{:.9f}".format(x_vals[max_ind0]))
        self.text10_1.setText("{:.3e}".format(error1[max_ind1]))
        self.text10_3.setText("{:.9f}".format(x_vals[max_ind1]))
        self.text11_1.setText("{:.3e}".format(error2[max_ind2]))
        self.text11_3.setText("{:.9f}".format(x_vals[max_ind2]))
        av = getA(n)
        bv = getB(n)
        cv = getC(n)
        dv = getD(n)
        for i in range(n):
            self.table1.insertRow(i)
            item1 = QTableWidgetItem()
            item1.setData(Qt.DisplayRole, "{:.9f}".format(x_vals[i]) )
            self.table1.setItem(i, 0, item1)
            item2 = QTableWidgetItem()
            item2.setData(Qt.DisplayRole, "{:.9f}".format(f[i]) )
            self.table1.setItem(i, 1, item2)
            item3 = QTableWidgetItem()
            item3.setData(Qt.DisplayRole, "{:.9f}".format(s[i]) )
            self.table1.setItem(i, 2, item3)
            item4 = QTableWidgetItem()
            item4.setData(Qt.DisplayRole, "{:.9f}".format(av[i]) )
            self.table1.setItem(i, 3, item4)
            item5 = QTableWidgetItem()
            item5.setData(Qt.DisplayRole, "{:.9f}".format(bv[i]) )
            self.table1.setItem(i, 4, item5)
            item6 = QTableWidgetItem()
            item6.setData(Qt.DisplayRole, "{:.9f}".format(cv[i]) )
            self.table1.setItem(i, 5, item6)
            item7 = QTableWidgetItem()
            item7.setData(Qt.DisplayRole, "{:.9f}".format(dv[i]) )
            self.table1.setItem(i, 6, item7)
            item8 = QTableWidgetItem()
            item8.setData(Qt.DisplayRole, "{:.3e}".format(error[i]) )
            self.table1.setItem(i, 7, item8)
            item9 = QTableWidgetItem()
            item9.setData(Qt.DisplayRole, "{:.3e}".format(error1[i]) )
            self.table1.setItem(i, 8, item9)
            item10 = QTableWidgetItem()
            item10.setData(Qt.DisplayRole, "{:.3e}".format(error2[i]) )
            self.table1.setItem(i, 9, item10)
        FreeAll() 





if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())