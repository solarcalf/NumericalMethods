import ctypes
import sys
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt

lib = ctypes.CDLL('./lib_dirichlet.so')
lib.solve_test_task.argtypes = [ctypes.c_uint8, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_double, ctypes.c_double]
lib.solve_test_custom_task.argtypes = [ctypes.c_uint8, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_double, ctypes.c_double]
lib.solve_main_task.argtypes = [ctypes.c_uint8, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_double, ctypes.c_double]

def test_task(solver_num:int, n: int, m:int, max_iterations:int, eps:float, omega:float):
    lib.solve_test_task(solver_num, n, m, max_iterations, eps, omega)

def test_custom_task(solver_num:int, n: int, m:int, max_iterations:int, eps:float, omega:float):
    lib.solve_test_custom_task(solver_num, n, m, max_iterations, eps, omega)

def main_task(solver_num:int, n: int, m:int, max_iterations:int, eps:float, omega:float):
    lib.solve_main_task(solver_num, n, m, max_iterations, eps, omega)



class Window(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Справка")
        self.setGeometry(100, 100, 1000, 850)


        #making tables
        self.table1 = QTableWidget(self)
        self.table1.setGeometry(360, 170, 630, 330)
        self.table1.setColumnCount(0)
        self.table1.setRowCount(0)

        self.table2 = QTableWidget(self)
        self.table2.setGeometry(360, 510, 630, 330)
        self.table2.setColumnCount(0)
        self.table2.setRowCount(0)



        #list of methods
        text1 = QtWidgets.QLabel(self)
        text1.setText("Метод")
        text1.setGeometry(85, 0, 100, 40)
        self.combo_box = QComboBox(self)
        self.combo_box.addItem("Метод верхней релаксации")
        self.combo_box.addItem("Метод минимальных невязок")
        self.combo_box.addItem("Метод Чебышёва")
        self.combo_box.addItem("Метод сопряженных градиентов")
        self.combo_box.setGeometry(10, 40, 210, 40)


        text2 = QtWidgets.QLabel(self)
        text2.setText("Задача")
        text2.setGeometry(85, 120, 100, 40)
        self.combo_box1 = QComboBox(self)
        self.combo_box1.addItem("Тестовая для стандартной области")
        self.combo_box1.addItem("Основная для стандартной области")
        self.combo_box1.addItem("Тестовая для нестандартной области")
        self.combo_box1.setGeometry(10, 165, 270, 35)


        text_omega = QtWidgets.QLabel(self)
        text_omega.setText("Омега (w) = ")
        text_omega.setGeometry(10, 90, 80, 30)
        self.omega = QLineEdit(self)
        self.omega.setGeometry(90, 90, 50, 30)
        self.omega.setText('1.0')

        set1 = QtWidgets.QLabel(self)
        set1.setText("Δu(x, y) = ")
        set1.setGeometry(10, 250, 80, 30)
        self.f = QtWidgets.QLabel(self)
        self.f.setGeometry(90, 250, 230, 30)
        self.f.setStyleSheet('background-color: white;')

        set2 = QtWidgets.QLabel(self)
        set2.setText(" при x ∈ ( -1,  1), y ∈ ( -1,  1);")
        set2.setGeometry(10, 280, 200, 30)

        set3 = QtWidgets.QLabel(self)
        set3.setText(" u ( -1, y) = ")
        set3.setGeometry(10, 310, 80, 30)
        self.mu1 = QtWidgets.QLabel(self)
        self.mu1.setGeometry(90, 310, 80, 30)
        self.mu1.setStyleSheet('background-color: white;')


        set4 = QtWidgets.QLabel(self)
        set4.setText(" u ( 1, y) = ")
        set4.setGeometry(170, 310, 80, 30)
        self.mu2 = QtWidgets.QLabel(self)
        self.mu2.setGeometry(240, 310, 80, 30)
        self.mu2.setStyleSheet('background-color: white;')


        self.set5 = QtWidgets.QLabel(self)
        self.set5.setText(" при y ∈ [ -1,  1];              при y ∈ [ -1,  1];")
        self.set5.setGeometry(40, 340, 350, 30)


        set6 = QtWidgets.QLabel(self)
        set6.setText(" u ( x, -1) = ")
        set6.setGeometry(10, 370, 80, 30)
        self.mu3 = QtWidgets.QLabel(self)
        self.mu3.setGeometry(90, 370, 80, 30)
        self.mu3.setStyleSheet('background-color: white;')


        set7 = QtWidgets.QLabel(self)
        set7.setText(" u ( x, 1) = ")
        set7.setGeometry(170, 370, 80, 30)
        self.mu4 = QtWidgets.QLabel(self)
        self.mu4.setGeometry(240, 370, 80, 30)
        self.mu4.setStyleSheet('background-color: white;')


        self.set8 = QtWidgets.QLabel(self)
        self.set8.setText(" при x ∈ [ -1,  1];              при x ∈ [ -1,  1];")
        self.set8.setGeometry(40, 400, 350, 30)

        set_line = QtWidgets.QLabel(self)
        set_line.setText("__________________________________________________________________________")
        set_line.setGeometry(30, 430, 300, 20)


        set_line = QtWidgets.QLabel(self)
        set_line.setText("Доп. ГУ для нестандартной сетки")
        set_line.setGeometry(50, 450, 250, 20)


        set9 = QtWidgets.QLabel(self)
        set9.setText(" u ( x, 0) = ")
        set9.setGeometry(10, 470, 80, 30)
        self.mu5 = QtWidgets.QLabel(self)
        self.mu5.setGeometry(90, 470, 80, 30)
        self.mu5.setStyleSheet('background-color: white;')


        set10 = QtWidgets.QLabel(self)
        set10.setText(" u ( 0, y) = ")
        set10.setGeometry(170, 470, 80, 30)
        self.mu6 = QtWidgets.QLabel(self)
        self.mu6.setGeometry(240, 470, 80, 30)
        self.mu6.setStyleSheet('background-color: white;')


        set11 = QtWidgets.QLabel(self)
        set11.setText(" при x ∈ [ -1,  0];")
        set11.setGeometry(40, 500, 120, 30)

        set11 = QtWidgets.QLabel(self)
        set11.setText(" при y ∈ [ -1,  0];")
        set11.setGeometry(170, 500, 120, 30)

        set_line1 = QtWidgets.QLabel(self)
        set_line1.setText("__________________________________________________________________________")
        set_line1.setGeometry(30, 530, 300, 20)

        setn = QtWidgets.QLabel(self)
        setn.setText("n = ")
        setn.setGeometry(10, 560, 30, 30)
        setm = QtWidgets.QLabel(self)
        setm.setText("m = ")
        setm.setGeometry(150, 560, 30, 30)
        self.line_editn = QLineEdit(self)
        self.line_editn.setGeometry(40, 560, 100, 30)
        self.line_editn.setText('8')
        self.line_editm = QLineEdit(self)
        self.line_editm.setGeometry(180, 560, 100, 30)
        self.line_editm.setText('8')

        self.report = QtWidgets.QLabel(self)
        self.report.setGeometry(230, 10, 760, 150)
        self.report.setStyleSheet('background-color: white;')


        setit = QtWidgets.QLabel(self)
        setit.setText("Максимальное количество итераций ")
        setit.setGeometry(10, 600, 240, 30)
        self.line_editit = QLineEdit(self)
        self.line_editit.setGeometry(250, 600, 100, 30)
        self.line_editit.setText('1000000')

        seteps = QtWidgets.QLabel(self)
        seteps.setText("Требуемая точность ")
        seteps.setGeometry(10, 635, 135, 30)
        self.line_editeps = QLineEdit(self)
        self.line_editeps.setGeometry(145, 635, 150, 30)
        self.line_editeps.setText('0.00000001')


        #button set task
        self.button1 = QtWidgets.QPushButton(self)
        self.button1.setGeometry(15, 210, 200, 30)
        self.button1.setText("Задать условия задачи")
        self.button1.clicked.connect(self.Set_task)


        #button solve task
        self.button2 = QtWidgets.QPushButton(self)
        self.button2.setGeometry(30, 760, 200, 30)
        self.button2.setText("Аппроксимировать")
        self.button2.clicked.connect(self.Solve_task)

    def Set_task(self):
        if(self.combo_box1.currentIndex() == 0):
            self.f.setText("4exp(1 - x^2 - y^2)(x^2 + y^2 - 1)")
            self.mu1.setText("exp(- y^2)")
            self.mu2.setText("exp(- y^2)")
            self.mu3.setText("exp(- x^2)")
            self.mu4.setText("exp(- x^2)")
            self.mu5.setText("")
            self.mu6.setText("")
            self.set5.setText(" при y ∈ [ -1,  1];              при y ∈ [ -1,  1];")
            self.set8.setText(" при x ∈ [ -1,  1];              при x ∈ [ -1,  1];")


        elif(self.combo_box1.currentIndex() == 1):
            self.f.setText("- |(sin(πxy))^3|")
            self.mu1.setText("1- y^2")
            self.mu2.setText("1 - y^2")
            self.mu3.setText("1 - x^2")
            self.mu4.setText("1 - x^2")
            self.mu5.setText("")
            self.mu6.setText("")
            self.set5.setText(" при y ∈ [ -1,  1];              при y ∈ [ -1,  1];")
            self.set8.setText(" при x ∈ [ -1,  1];              при x ∈ [ -1,  1];")
        else:
            self.f.setText("4exp(1 - x^2 - y^2)(x^2 + y^2 - 1)")
            self.mu1.setText("exp(- y^2)")
            self.mu2.setText("exp(- y^2)")
            self.mu3.setText("exp(- x^2)")
            self.mu4.setText("exp(- x^2)")
            self.mu5.setText("exp(1 - x^2)")
            self.mu6.setText("exp(1 - y^2)")
            self.set5.setText(" при y ∈ [ 0,  1];               при y ∈ [ -1,  1];")
            self.set8.setText(" при x ∈ [ 0,  1];               при x ∈ [ -1,  1];")
 

    def Solve_task(self):
        n = (int)(self.line_editn.text())
        m = (int)(self.line_editm.text())
        omega = (float)(self.omega.text())
        max_iterations = (int)(self.line_editit.text())
        eps = (float)(self.line_editeps.text())
        task_num = self.combo_box1.currentIndex()
        solver_num = (int)(self.combo_box.currentIndex())
        solver_str = " "
        if(task_num == 0):
            solver_str += "Тестовая задача на стандартной сетке решена "
        elif(task_num == 1):
            solver_str += "Основная задача на стандартной сетке решена "
        else:
            solver_str += "Тестовая задача на нестандартной сетке решена "
        
        if(solver_num == 0):
            solver_str += "методом верхней релаксации с параметром w = "+ self.omega.text()
        elif(solver_num == 1):
            solver_str += "методом минимальных невязок"
        elif(solver_num == 2):
            solver_str += "методом Чебышева"
        else:
            solver_str += "методом сопряженных градиентов"
        
        if(task_num == 0):
            test_task(solver_num, n, m, max_iterations, eps, omega)

            error = open("../files/Error.txt", 'r')
            solver_results = open("../files/Solver_results.txt", 'r')
            results = [row.strip() for row in solver_results]

            self.report.setText("Для решения тестовой задачи использованы сетка с числом разбиений\n по х : n = " + (str)(n)+
            ", и числом разбиений по y: m = "+(str)(m)+",\n " + solver_str + ", \n применены критерии остановки по точности eps(мет) = "+
            (str)(eps)+" и по числу итераций N(max) = "+ (str)(max_iterations)+".\n \n " + "На решение схемы (СЛАУ) затрачено " + (str)(results[2]) + 
            " итераций и достигнута точность " + (str)(results[0]) + "\nСхема решена с невязкой " + (str)(results[1]) + "\n" + error.read())
            error.close()
            solver_results.close()


        elif(task_num == 1):
            main_task(solver_num, n, m, max_iterations, eps, omega)

            error = open("../files/Error.txt", 'r')
            solver_results = open("../files/Solver_results.txt", 'r')
            results = [row.strip() for row in solver_results]

            self.report.setText("Для решения основной задачи использованы сетка с числом разбиений\n по х : n = " + (str)(n)+
            ", и числом разбиений по y: m = "+(str)(m)+",\n " + solver_str + ", \n применены критерии остановки по точности eps(мет) = "+
            (str)(eps)+" и по числу итераций N(max) = "+ (str)(max_iterations)+".\n \n " + "На решение схемы (СЛАУ) затрачено " + (str)(results[2]) + 
            " итераций и достигнута точность " + (str)(results[0]) + "\nСхема решена с невязкой " + (str)(results[1]) + "\n" + error.read())
            error.close()
            solver_results.close()

        else:
            test_custom_task(solver_num, n, m, max_iterations, eps, omega)

            error = open("../files/Error.txt", 'r')
            solver_results = open("../files/Solver_results.txt", 'r')
            results = [row.strip() for row in solver_results]

            self.report.setText("Для решения тестовой задачи использованы сетка с числом разбиений\n по х : n = " + (str)(n)+
            ", и числом разбиений по y: m = "+(str)(m)+",\n " + solver_str + ", \n применены критерии остановки по точности eps(мет) = "+
            (str)(eps)+" и по числу итераций N(max) = "+ (str)(max_iterations)+".\n \n " + "На решение схемы (СЛАУ) затрачено " + (str)(results[2]) + 
            " итераций и достигнута точность " + (str)(results[0]) + "\nСхема решена с невязкой " + (str)(results[1]) + "\n" + error.read())
            error.close()
            solver_results.close()
        self.load_data_to_table1("../files/Approximation.txt")
        self.load_data_to_table2("../files/Correct.txt")

    def load_data_to_table1(self, filename):
        self.table1.setRowCount(0)
        self.table1.setColumnCount(0)
        try:
            with open(filename, 'r') as file:
                lines = file.readlines()

            # Определяем количество строк и столбцов
            num_rows = len(lines)
            num_cols = max(len(line.split()) for line in lines)

            # Устанавливаем количество строк и столбцов в таблице
            self.table1.setRowCount(num_rows)
            self.table1.setColumnCount(num_cols)

            # Заполняем таблицу данными
            for row_index, line in enumerate(lines):
                values = line.split()
                for col_index, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    self.table1.setItem(row_index, col_index, item)
        except FileNotFoundError:
            print("Файл не найден")
        except Exception as e:
            print(f"Ошибка при чтении файла: {e}")

    def load_data_to_table2(self, filename):
        self.table2.setRowCount(0)
        self.table2.setColumnCount(0)
        try:
            with open(filename, 'r') as file:
                lines = file.readlines()

            # Определяем количество строк и столбцов
            num_rows = len(lines)
            num_cols = max(len(line.split()) for line in lines)

            # Устанавливаем количество строк и столбцов в таблице
            self.table2.setRowCount(num_rows)
            self.table2.setColumnCount(num_cols)

            # Заполняем таблицу данными
            for row_index, line in enumerate(lines):
                values = line.split()
                for col_index, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    self.table2.setItem(row_index, col_index, item)
        except FileNotFoundError:
            print("Файл не найден")
        except Exception as e:
            print(f"Ошибка при чтении файла: {e}")



        


        


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())