#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据分析与方程建模软件
功能：
1. 导入CSV表格数据
2. 手动输入数据
3. 自定义关联数据（将方程参数映射到数据列）
4. 输入方程（包括常见方程和偏微分方程）
5. 进行建模运算
6. 输出运算结果或绘图
7. 保存常用方程
8. 预设常见方程
"""

import sys
import json
import os
import re
import numpy as np
import pandas as pd
from scipy import integrate, optimize, interpolate
from scipy.integrate import odeint, solve_ivp
import sympy as sp
from sympy import symbols, diff, integrate as sym_integrate, lambdify, sympify, Function, Eq, dsolve

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTableWidget, QTableWidgetItem,
    QTabWidget, QComboBox, QTextEdit, QFileDialog, QMessageBox,
    QSplitter, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QListWidget, QListWidgetItem, QDialog, QDialogButtonBox,
    QHeaderView, QMenuBar, QMenu, QToolBar, QStatusBar, QFrame,
    QScrollArea, QCheckBox, QGridLayout, QInputDialog
)
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QFont, QIcon, QKeySequence

import matplotlib

matplotlib.use('Qt5Agg')  # ✅ 修复：正确的后端名称
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


def resource_path(relative_path):
    """获取资源绝对路径（兼容开发环境和打包后）"""
    try:
        # PyInstaller 创建临时文件夹
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# 使用示例：加载图标或数据文件
# icon_path = resource_path("assets/icon.png")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 预设方程库
PRESET_EQUATIONS = {
    "线性回归": {
        "equation": "y = a * x + b",
        "params": ["a", "b"],
        "description": "一元线性回归方程 y = ax + b"
    },
    "多项式": {
        "equation": "y = a * x**2 + b * x + c",
        "params": ["a", "b", "c"],
        "description": "二次多项式方程"
    },
    "指数增长": {
        "equation": "y = a * exp(b * x)",
        "params": ["a", "b"],
        "description": "指数增长模型"
    },
    "对数函数": {
        "equation": "y = a * log(x) + b",
        "params": ["a", "b"],
        "description": "对数函数模型"
    },
    "幂函数": {
        "equation": "y = a * x**b",
        "params": ["a", "b"],
        "description": "幂函数模型"
    },
    "逻辑斯蒂": {
        "equation": "y = L / (1 + exp(-k * (x - x0)))",
        "params": ["L", "k", "x0"],
        "description": "逻辑斯蒂增长模型（S型曲线）"
    },
    "正弦函数": {
        "equation": "y = A * sin(omega * x + phi) + C",
        "params": ["A", "omega", "phi", "C"],
        "description": "正弦波函数"
    },
    "高斯分布": {
        "equation": "y = A * exp(-(x - mu)**2 / (2 * sigma**2))",
        "params": ["A", "mu", "sigma"],
        "description": "高斯（正态）分布"
    },
    "一阶ODE": {
        "equation": "dy/dt = -k * y",
        "params": ["k"],
        "description": "一阶线性常微分方程（衰减模型）"
    },
    "二阶ODE": {
        "equation": "d2y/dt2 + 2*zeta*omega*dy/dt + omega**2*y = 0",
        "params": ["zeta", "omega"],
        "description": "二阶线性常微分方程（阻尼振动）"
    },
    "热传导方程": {
        "equation": "du/dt = alpha * d2u/dx2",
        "params": ["alpha"],
        "description": "一维热传导方程（偏微分方程）"
    },
    "波动方程": {
        "equation": "d2u/dt2 = c**2 * d2u/dx2",
        "params": ["c"],
        "description": "一维波动方程（偏微分方程）"
    },
    "反应扩散": {
        "equation": "du/dt = D * d2u/dx2 + r * u * (1 - u/K)",
        "params": ["D", "r", "K"],
        "description": "反应扩散方程（Fisher-KPP方程）"
    }
}


class MplCanvas(FigureCanvas):
    """Matplotlib画布类"""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)


class EquationDialog(QDialog):
    """方程编辑对话框"""

    def __init__(self, parent=None, equation_name="", equation_data=None):
        super().__init__(parent)
        self.setWindowTitle("编辑方程")
        self.setMinimumWidth(500)
        self.equation_data = equation_data or {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # 方程名称
        form_layout = QFormLayout()
        self.name_edit = QLineEdit(self.equation_data.get("name", ""))
        form_layout.addRow("方程名称:", self.name_edit)

        # 方程表达式
        self.eq_edit = QLineEdit(self.equation_data.get("equation", ""))
        self.eq_edit.setPlaceholderText("例如: y = a * x + b")
        form_layout.addRow("方程表达式:", self.eq_edit)

        # 参数列表
        self.params_edit = QLineEdit(", ".join(self.equation_data.get("params", [])))
        self.params_edit.setPlaceholderText("例如: a, b, c")
        form_layout.addRow("参数 (逗号分隔):", self.params_edit)

        # 描述
        self.desc_edit = QTextEdit(self.equation_data.get("description", ""))
        self.desc_edit.setMaximumHeight(80)
        form_layout.addRow("描述:", self.desc_edit)

        layout.addLayout(form_layout)

        # 按钮
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_data(self):
        return {
            "name": self.name_edit.text(),
            "equation": self.eq_edit.text(),
            "params": [p.strip() for p in self.params_edit.text().split(",") if p.strip()],
            "description": self.desc_edit.toPlainText()
        }


class DataTableWidget(QWidget):
    """数据表格组件"""
    data_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataframe = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # 工具栏
        toolbar = QHBoxLayout()

        self.import_btn = QPushButton("导入CSV")
        self.import_btn.setToolTip("从CSV文件导入数据")
        self.import_btn.clicked.connect(self.import_csv)
        toolbar.addWidget(self.import_btn)

        self.export_btn = QPushButton("导出CSV")
        self.export_btn.setToolTip("导出数据到CSV文件")
        self.export_btn.clicked.connect(self.export_csv)
        toolbar.addWidget(self.export_btn)

        toolbar.addSpacing(20)

        self.add_row_btn = QPushButton("添加行")
        self.add_row_btn.clicked.connect(self.add_row)
        toolbar.addWidget(self.add_row_btn)

        self.add_col_btn = QPushButton("添加列")
        self.add_col_btn.clicked.connect(self.add_column)
        toolbar.addWidget(self.add_col_btn)

        self.del_row_btn = QPushButton("删除行")
        self.del_row_btn.clicked.connect(self.delete_row)
        toolbar.addWidget(self.del_row_btn)

        self.del_col_btn = QPushButton("删除列")
        self.del_col_btn.clicked.connect(self.delete_column)
        toolbar.addWidget(self.del_col_btn)

        toolbar.addStretch()

        self.clear_btn = QPushButton("清空")
        self.clear_btn.clicked.connect(self.clear_data)
        toolbar.addWidget(self.clear_btn)

        layout.addLayout(toolbar)

        # 数据表格
        self.table = QTableWidget()
        self.table.setColumnCount(0)
        self.table.setRowCount(0)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.itemChanged.connect(self.on_item_changed)
        layout.addWidget(self.table)

        # 信息显示
        self.info_label = QLabel("暂无数据")
        layout.addWidget(self.info_label)

    def import_csv(self):
        """导入CSV文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择CSV文件", "", "CSV文件 (*.csv);;所有文件 (*.*)"
        )
        if file_path:
            try:
                # 尝试不同编码
                encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
                for encoding in encodings:
                    try:
                        self.dataframe = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise Exception("无法解码文件，请检查编码格式")

                self.update_table_from_df()
                self.info_label.setText(
                    f"已加载: {file_path} | 行数: {len(self.dataframe)} | 列数: {len(self.dataframe.columns)}")
                self.data_changed.emit()
                QMessageBox.information(self, "成功", f"成功导入 {len(self.dataframe)} 行数据")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导入失败: {str(e)}")

    def export_csv(self):
        """导出CSV文件"""
        if self.dataframe is None or self.dataframe.empty:
            QMessageBox.warning(self, "警告", "没有数据可导出")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存CSV文件", "", "CSV文件 (*.csv);;所有文件 (*.*)"
        )
        if file_path:
            try:
                self.dataframe.to_csv(file_path, index=False, encoding='utf-8-sig')
                QMessageBox.information(self, "成功", "数据导出成功")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")

    def update_table_from_df(self):
        """从DataFrame更新表格"""
        if self.dataframe is None:
            return

        self.table.blockSignals(True)
        self.table.setRowCount(len(self.dataframe))
        self.table.setColumnCount(len(self.dataframe.columns))
        self.table.setHorizontalHeaderLabels(list(self.dataframe.columns))

        for i in range(len(self.dataframe)):
            for j in range(len(self.dataframe.columns)):
                value = self.dataframe.iloc[i, j]
                if pd.isna(value):
                    item = QTableWidgetItem("")
                else:
                    item = QTableWidgetItem(str(value))
                self.table.setItem(i, j, item)

        self.table.blockSignals(False)

    def update_df_from_table(self):
        """从表格更新DataFrame"""
        rows = self.table.rowCount()
        cols = self.table.columnCount()

        if rows == 0 or cols == 0:
            self.dataframe = None
            return

        headers = []
        for j in range(cols):
            header_item = self.table.horizontalHeaderItem(j)
            headers.append(header_item.text() if header_item else f"列{j + 1}")

        data = []
        for i in range(rows):
            row_data = []
            for j in range(cols):
                item = self.table.item(i, j)
                value = item.text() if item else ""
                # 尝试转换为数值
                try:
                    value = float(value) if value else np.nan
                except:
                    pass
                row_data.append(value)
            data.append(row_data)

        self.dataframe = pd.DataFrame(data, columns=headers)

    def on_item_changed(self):
        """表格项改变时更新DataFrame"""
        self.update_df_from_table()
        self.data_changed.emit()

    def add_row(self):
        """添加行"""
        row = self.table.rowCount()
        self.table.insertRow(row)
        if self.dataframe is not None:
            self.update_df_from_table()
            self.data_changed.emit()

    def add_column(self):
        """添加列"""
        col = self.table.columnCount()
        self.table.insertColumn(col)
        self.table.setHorizontalHeaderItem(col, QTableWidgetItem(f"列{col + 1}"))
        if self.dataframe is not None:
            self.update_df_from_table()
            self.data_changed.emit()

    def delete_row(self):
        """删除选中行"""
        current_row = self.table.currentRow()
        if current_row >= 0:
            self.table.removeRow(current_row)
            self.update_df_from_table()
            self.data_changed.emit()

    def delete_column(self):
        """删除选中列"""
        current_col = self.table.currentColumn()
        if current_col >= 0:
            self.table.removeColumn(current_col)
            self.update_df_from_table()
            self.data_changed.emit()

    def clear_data(self):
        """清空数据"""
        reply = QMessageBox.question(
            self, "确认", "确定要清空所有数据吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.dataframe = None
            self.info_label.setText("暂无数据")
            self.data_changed.emit()

    def get_dataframe(self):
        """获取当前DataFrame"""
        self.update_df_from_table()
        return self.dataframe

    def get_column_names(self):
        """获取列名列表"""
        cols = []
        for j in range(self.table.columnCount()):
            item = self.table.horizontalHeaderItem(j)
            cols.append(item.text() if item else f"列{j + 1}")
        return cols

    def get_column_data(self, col_name):
        """获取指定列的数据"""
        if self.dataframe is not None and col_name in self.dataframe.columns:
            return self.dataframe[col_name].values
        return None


class EquationManagerWidget(QWidget):
    """方程管理组件"""
    equation_selected = pyqtSignal(str, dict)  # 发射方程名称和数据

    def __init__(self, parent=None):
        super().__init__(parent)
        self.user_equations = {}
        self.load_user_equations()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # 标签页
        self.tab_widget = QTabWidget()

        # 预设方程页
        self.preset_widget = QWidget()
        preset_layout = QVBoxLayout(self.preset_widget)

        self.preset_list = QListWidget()
        for name, data in PRESET_EQUATIONS.items():
            item = QListWidgetItem(f"{name}\n{data['equation']}")
            item.setData(Qt.ItemDataRole.UserRole, {"name": name, **data})
            item.setToolTip(data.get("description", ""))
            self.preset_list.addItem(item)

        self.preset_list.itemDoubleClicked.connect(self.on_preset_selected)
        self.preset_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        preset_layout.addWidget(QLabel("双击选择方程:"))
        preset_layout.addWidget(self.preset_list)

        self.use_preset_btn = QPushButton("使用选中方程")
        self.use_preset_btn.clicked.connect(self.use_selected_preset)
        preset_layout.addWidget(self.use_preset_btn)

        self.tab_widget.addTab(self.preset_widget, "预设方程")

        # 用户方程页
        self.user_widget = QWidget()
        user_layout = QVBoxLayout(self.user_widget)

        self.user_list = QListWidget()
        self.refresh_user_list()
        self.user_list.itemDoubleClicked.connect(self.on_user_selected)
        user_layout.addWidget(QLabel("我的方程 (双击选择):"))
        user_layout.addWidget(self.user_list)

        user_btn_layout = QHBoxLayout()

        self.add_eq_btn = QPushButton("添加方程")
        self.add_eq_btn.clicked.connect(self.add_equation)
        user_btn_layout.addWidget(self.add_eq_btn)

        self.edit_eq_btn = QPushButton("编辑")
        self.edit_eq_btn.clicked.connect(self.edit_equation)
        user_btn_layout.addWidget(self.edit_eq_btn)

        self.del_eq_btn = QPushButton("删除")
        self.del_eq_btn.clicked.connect(self.delete_equation)
        user_btn_layout.addWidget(self.del_eq_btn)

        self.use_user_btn = QPushButton("使用选中方程")
        self.use_user_btn.clicked.connect(self.use_selected_user)
        user_btn_layout.addWidget(self.use_user_btn)

        user_layout.addLayout(user_btn_layout)

        self.tab_widget.addTab(self.user_widget, "我的方程")

        layout.addWidget(self.tab_widget)

        # 当前方程显示
        self.current_eq_group = QGroupBox("当前方程")
        eq_layout = QVBoxLayout(self.current_eq_group)

        self.current_eq_label = QLabel("未选择方程")
        self.current_eq_label.setWordWrap(True)
        self.current_eq_label.setStyleSheet("font-size: 14px; color: #0066cc;")
        eq_layout.addWidget(self.current_eq_label)

        self.current_params_label = QLabel("")
        self.current_params_label.setWordWrap(True)
        eq_layout.addWidget(self.current_params_label)

        layout.addWidget(self.current_eq_group)

        # 方程输入区
        input_group = QGroupBox("自定义方程输入")
        input_layout = QVBoxLayout(input_group)

        self.custom_eq_input = QLineEdit()
        self.custom_eq_input.setPlaceholderText("直接输入方程，例如: y = a * x + b")
        input_layout.addWidget(self.custom_eq_input)

        custom_btn_layout = QHBoxLayout()

        self.parse_eq_btn = QPushButton("解析方程")
        self.parse_eq_btn.clicked.connect(self.parse_custom_equation)
        custom_btn_layout.addWidget(self.parse_eq_btn)

        self.save_eq_btn = QPushButton("保存到我的方程")
        self.save_eq_btn.clicked.connect(self.save_custom_equation)
        custom_btn_layout.addWidget(self.save_eq_btn)

        input_layout.addLayout(custom_btn_layout)
        layout.addWidget(input_group)

        # 帮助信息
        help_text = """
        <b>方程输入说明:</b><br>
        • 使用 <b>x</b> 作为自变量，<b>y</b> 作为因变量<br>
        • 参数使用字母表示，如 a, b, c<br>
        • 支持运算符: +, -, *, /, **, (, )<br>
        • 支持函数: sin, cos, exp, log, sqrt等<br>
        • ODE使用 dy/dt 或 y' 表示导数<br>
        • PDE使用 du/dt, du/dx, d2u/dx2 等表示偏导<br>
        •请将常数值一并在方程中输入，不支持后续赋值
        """
        help_label = QLabel(help_text)
        help_label.setWordWrap(True)
        help_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        layout.addWidget(help_label)

        layout.addStretch()

        self.current_equation = None

    def load_user_equations(self):
        """加载用户保存的方程"""
        config_path = os.path.join(os.path.expanduser("~"), ".data_modeling_app")
        eq_file = os.path.join(config_path, "equations.json")
        if os.path.exists(eq_file):
            try:
                with open(eq_file, 'r', encoding='utf-8') as f:
                    self.user_equations = json.load(f)
            except:
                self.user_equations = {}

    def save_user_equations(self):
        """保存用户方程"""
        config_path = os.path.join(os.path.expanduser("~"), ".data_modeling_app")
        os.makedirs(config_path, exist_ok=True)
        eq_file = os.path.join(config_path, "equations.json")
        try:
            with open(eq_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_equations, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.warning(self, "警告", f"保存方程失败: {str(e)}")

    def refresh_user_list(self):
        """刷新用户方程列表"""
        self.user_list.clear()
        for name, data in self.user_equations.items():
            item = QListWidgetItem(f"{name}\n{data['equation']}")
            item.setData(Qt.ItemDataRole.UserRole, {"name": name, **data})
            item.setToolTip(data.get("description", ""))
            self.user_list.addItem(item)

    def on_preset_selected(self, item):
        """预设方程被双击"""
        data = item.data(Qt.ItemDataRole.UserRole)
        self.set_current_equation(data)

    def on_user_selected(self, item):
        """用户方程被双击"""
        data = item.data(Qt.ItemDataRole.UserRole)
        self.set_current_equation(data)

    def use_selected_preset(self):
        """使用选中的预设方程"""
        item = self.preset_list.currentItem()
        if item:
            data = item.data(Qt.ItemDataRole.UserRole)
            self.set_current_equation(data)
        else:
            QMessageBox.information(self, "提示", "请先选择一个方程")

    def use_selected_user(self):
        """使用选中的用户方程"""
        item = self.user_list.currentItem()
        if item:
            data = item.data(Qt.ItemDataRole.UserRole)
            self.set_current_equation(data)
        else:
            QMessageBox.information(self, "提示", "请先选择一个方程")

    def set_current_equation(self, data):
        """设置当前方程"""
        self.current_equation = data
        self.current_eq_label.setText(f"<b>{data['name']}</b><br>{data['equation']}")
        params_str = ", ".join(data.get("params", []))
        self.current_params_label.setText(f"参数: {params_str}")
        self.equation_selected.emit(data["name"], data)

    def add_equation(self):
        """添加新方程"""
        dialog = EquationDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_data()
            if data["name"] and data["equation"]:
                self.user_equations[data["name"]] = {
                    "equation": data["equation"],
                    "params": data["params"],
                    "description": data["description"]
                }
                self.save_user_equations()
                self.refresh_user_list()
            else:
                QMessageBox.warning(self, "警告", "方程名称和表达式不能为空")

    def edit_equation(self):
        """编辑方程"""
        item = self.user_list.currentItem()
        if item:
            data = item.data(Qt.ItemDataRole.UserRole)
            dialog = EquationDialog(self, data["name"], data)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                new_data = dialog.get_data()
                # 如果名称改变，删除旧的
                if new_data["name"] != data["name"]:
                    del self.user_equations[data["name"]]
                self.user_equations[new_data["name"]] = {
                    "equation": new_data["equation"],
                    "params": new_data["params"],
                    "description": new_data["description"]
                }
                self.save_user_equations()
                self.refresh_user_list()
        else:
            QMessageBox.information(self, "提示", "请先选择一个方程")

    def delete_equation(self):
        """删除方程"""
        item = self.user_list.currentItem()
        if item:
            data = item.data(Qt.ItemDataRole.UserRole)
            reply = QMessageBox.question(
                self, "确认", f"确定要删除方程 '{data['name']}' 吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                del self.user_equations[data["name"]]
                self.save_user_equations()
                self.refresh_user_list()
        else:
            QMessageBox.information(self, "提示", "请先选择一个方程")

    def parse_custom_equation(self):
        """解析自定义方程"""
        eq_str = self.custom_eq_input.text().strip()
        if not eq_str:
            QMessageBox.warning(self, "警告", "请输入方程")
            return

        try:
            # 尝试解析方程，提取参数
            params = self.extract_params(eq_str)
            data = {
                "name": "自定义方程",
                "equation": eq_str,
                "params": params,
                "description": "用户自定义方程"
            }
            self.set_current_equation(data)
            QMessageBox.information(self, "成功", f"方程解析成功！\n检测到参数: {', '.join(params)}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"方程解析失败: {str(e)}")

    def extract_params(self, eq_str):
        """从方程中提取参数"""
        # 移除等号右边部分
        if "=" in eq_str:
            _, right = eq_str.split("=", 1)
        else:
            right = eq_str

        # 定义已知的数学函数和变量
        known = {'x', 'y', 't', 'u', 'sin', 'cos', 'tan', 'exp', 'log', 'ln',
                 'sqrt', 'pi', 'e', 'abs', 'max', 'min', 'dy', 'dt', 'du', 'dx'}

        # 提取所有单词
        words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', right)

        # 过滤出参数（排除已知函数和变量）
        params = []
        for w in words:
            if w not in known and w not in params:
                params.append(w)

        return params

    def save_custom_equation(self):
        """保存自定义方程"""
        if self.current_equation is None:
            QMessageBox.warning(self, "警告", "请先解析或选择一个方程")
            return

        name, ok = QInputDialog.getText(self, "保存方程", "请输入方程名称:")
        if ok and name:
            self.user_equations[name] = {
                "equation": self.current_equation["equation"],
                "params": self.current_equation["params"],
                "description": self.current_equation.get("description", "")
            }
            self.save_user_equations()
            self.refresh_user_list()
            QMessageBox.information(self, "成功", "方程已保存")

    def get_current_equation(self):
        """获取当前选中的方程"""
        return self.current_equation


class ParameterMappingWidget(QWidget):
    """参数关联组件"""
    mapping_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.column_names = []
        self.param_widgets = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # 说明标签
        info_label = QLabel("将方程参数与数据列关联，或输入常数值")
        info_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(info_label)

        # 参数映射区域（带滚动条）
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.params_container = QWidget()
        self.params_layout = QFormLayout(self.params_container)
        self.params_layout.setSpacing(10)

        scroll.setWidget(self.params_container)
        layout.addWidget(scroll)

        # 快速设置按钮
        quick_layout = QHBoxLayout()

        self.clear_mapping_btn = QPushButton("清空所有映射")
        self.clear_mapping_btn.clicked.connect(self.clear_mapping)
        quick_layout.addWidget(self.clear_mapping_btn)

        self.auto_map_btn = QPushButton("自动映射")
        self.auto_map_btn.setToolTip("尝试根据名称自动匹配参数和列")
        self.auto_map_btn.clicked.connect(self.auto_mapping)
        quick_layout.addWidget(self.auto_map_btn)

        quick_layout.addStretch()
        layout.addLayout(quick_layout)

        layout.addStretch()

    def set_columns(self, column_names):
        """设置可用的数据列"""
        self.column_names = column_names
        # 更新现有参数控件中的列选择下拉框
        for param, widgets in self.param_widgets.items():
            type_combo = widgets["type_combo"]
            value_widget = widgets["value_widget"]

            # 如果当前是数据列类型，刷新选项
            if type_combo.currentIndex() == 1 and isinstance(value_widget, QComboBox):
                current_text = value_widget.currentText()
                value_widget.clear()
                value_widget.addItems(self.column_names)
                # 保留之前的选择（如果可用）
                if current_text in self.column_names:
                    value_widget.setCurrentText(current_text)

        self.mapping_changed.emit()

    def set_parameters(self, param_names):
        """设置需要映射的参数"""
        # 清除旧的参数控件
        for i in reversed(range(self.params_layout.rowCount())):
            self.params_layout.removeRow(i)
        self.param_widgets = {}

        # 创建新的参数映射控件
        for param in param_names:
            widget = QWidget()
            h_layout = QHBoxLayout(widget)
            h_layout.setContentsMargins(0, 0, 0, 0)

            # 类型选择
            type_combo = QComboBox()
            type_combo.addItems(["数据列", "变量x", "变量t"])
            type_combo.setCurrentIndex(0)  # 默认选择数据列
            type_combo.currentIndexChanged.connect(lambda: self.on_type_changed(param))
            h_layout.addWidget(type_combo)

            # 值输入/列选择
            value_widget = QComboBox()
            value_widget.addItems(self.column_names)
            h_layout.addWidget(value_widget, 1)

            self.param_widgets[param] = {
                "type_combo": type_combo,
                "value_widget": value_widget
            }

            self.params_layout.addRow(f"{param}:", widget)

        self.mapping_changed.emit()

    def on_type_changed(self, param):
        """参数类型改变时更新控件"""
        widgets = self.param_widgets[param]
        type_idx = widgets["type_combo"].currentIndex()

        # 获取父容器和布局
        container = widgets["type_combo"].parent()
        layout = container.layout()

        # 移除旧的值控件
        old_value_widget = widgets["value_widget"]
        layout.removeWidget(old_value_widget)
        old_value_widget.deleteLater()

        # 创建新的值控件

        if type_idx == 0:  # 数据列
            new_widget = QComboBox()
            new_widget.addItems(self.column_names)
            new_widget.setFixedWidth(120)
        elif type_idx == 1:  # 变量x
            new_widget = QLabel("变量x")
            new_widget.setStyleSheet("color: gray;")
            new_widget.setFixedWidth(120)
        else:  # 变量t
            new_widget = QLabel("变量t")
            new_widget.setStyleSheet("color: gray;")
            new_widget.setFixedWidth(120)

        # 添加到布局
        layout.addWidget(new_widget)
        widgets["value_widget"] = new_widget

        # 强制刷新界面
        container.update()
        self.mapping_changed.emit()

    def clear_mapping(self):
        """清空所有映射"""
        for param, widgets in self.param_widgets.items():
            widgets["type_combo"].setCurrentIndex(0)

    def auto_mapping(self):
        """尝试自动映射参数到列"""
        for param, widgets in self.param_widgets.items():
            type_combo = widgets["type_combo"]
            value_widget = widgets["value_widget"]

            # 如果参数名匹配列名，选择该列
            if param in self.column_names:
                type_combo.setCurrentIndex(0)  # 数据列
                if isinstance(value_widget, QComboBox):
                    value_widget.setCurrentText(param)

    def get_parameter_values(self, dataframe=None, x_values=None, t_values=None):
        """获取参数值字典"""
        param_values = {}

        for param, widgets in self.param_widgets.items():
            type_idx = widgets["type_combo"].currentIndex()
            value_widget = widgets["value_widget"]

            if type_idx == 0:  # 数据列
                if dataframe is not None and isinstance(value_widget, QComboBox):
                    col_name = value_widget.currentText()
                    if col_name in dataframe.columns:
                        param_values[param] = dataframe[col_name].values
                    else:
                        param_values[param] = None
                else:
                    param_values[param] = None
            elif type_idx == 1:  # 变量x
                param_values[param] = x_values
            elif type_idx == 2:  # 变量t
                param_values[param] = t_values

        return param_values

    def get_mapping_info(self):
        """获取映射信息（用于显示）"""
        info = []
        for param, widgets in self.param_widgets.items():
            type_idx = widgets["type_combo"].currentIndex()
            value_widget = widgets["value_widget"]

            type_names = ["数据列", "变量x", "变量t"]
            type_name = type_names[type_idx]

            if type_idx == 0 and isinstance(value_widget, QComboBox):
                value = value_widget.currentText()
            else:
                value = type_name

            info.append(f"{param} = {value}")

        return info


class SolverThread(QThread):
    """求解线程（防止界面卡顿）"""
    finished = pyqtSignal(object, str)  # 结果和消息
    progress = pyqtSignal(str)  # 进度消息

    def __init__(self, equation_data, param_values, dataframe, x_range, t_range):
        super().__init__()
        self.equation_data = equation_data
        self.param_values = param_values
        self.dataframe = dataframe
        self.x_range = x_range
        self.t_range = t_range
        self.is_pde = False

    def run(self):
        try:
            self.progress.emit("正在解析方程...")
            result = self.solve()
            self.finished.emit(result, "计算完成")
        except Exception as e:
            import traceback
            error_msg = f"计算错误: {str(e)}\n{traceback.format_exc()}"
            self.finished.emit(None, error_msg)

    def solve(self):
        """求解方程"""
        eq_str = self.equation_data["equation"]
        params = self.equation_data.get("params", [])

        # 检测方程类型
        if "d2u" in eq_str or "du/d" in eq_str:
            return self.solve_pde()
        elif "dy/d" in eq_str or "y'" in eq_str or "d2y" in eq_str:
            return self.solve_ode()
        else:
            return self.solve_algebraic()

    def solve_algebraic(self):
        """求解代数方程"""
        eq_str = self.equation_data["equation"]

        # 解析方程
        if "=" in eq_str:
            left, right = eq_str.split("=", 1)
        else:
            right = eq_str
            left = "y"

        # 准备参数 - 所有参数都必须是数值
        param_dict = {}
        x_values = None

        for param, value in self.param_values.items():
            if value is not None:
                if isinstance(value, np.ndarray):
                    if param == 'x':
                        x_values = value
                    else:
                        # 如果是数组，取第一个值作为常数
                        param_dict[param] = float(value[0]) if len(value) > 0 else 0.0
                else:
                    # 确保是数值
                    param_dict[param] = float(value)
            else:
                # 参数为None，使用默认值0
                param_dict[param] = 0.0

        # 如果没有x值，生成默认范围
        if x_values is None:
            if self.x_range:
                x_values = np.linspace(self.x_range[0], self.x_range[1], 100)
            elif self.dataframe is not None and len(self.dataframe) > 0:
                x_values = np.arange(len(self.dataframe))
            else:
                x_values = np.linspace(0, 10, 100)

        # 确保 x_values 是 numpy 数组
        x_values = np.array(x_values, dtype=float)

        try:
            # 直接用 eval 计算（更可靠）
            # 创建安全的计算环境
            safe_dict = {
                'x': x_values,
                'np': np,
                'sin': np.sin,
                'cos': np.cos,
                'tan': np.tan,
                'exp': np.exp,
                'log': np.log,
                'ln': np.log,
                'sqrt': np.sqrt,
                'abs': np.abs,
                'pi': np.pi,
                'e': np.e,
            }
            # 添加所有参数
            safe_dict.update(param_dict)

            # 替换 ^ 为 **（Python语法）
            expr_str = right.strip().replace('^', '**')

            # 计算
            y_values = eval(expr_str, {"__builtins__": {}}, safe_dict)

            # 确保结果是 numpy 数组
            y_values = np.array(y_values, dtype=float)

            # 如果结果是标量，扩展为数组
            if y_values.shape == ():
                y_values = np.full_like(x_values, y_values)
            elif len(y_values) == 1:
                y_values = np.full_like(x_values, y_values[0])

            return {
                'type': 'algebraic',
                'x': x_values,
                'y': y_values,
                'equation': eq_str
            }
        except Exception as e:
            raise Exception(f"方程求解失败: {str(e)}")

    def solve_ode(self):
        """求解常微分方程"""
        eq_str = self.equation_data["equation"]

        # 简化ODE解析
        # 这里使用简化的实现，实际应用需要更复杂的解析

        # 获取参数值
        k = self.param_values.get('k', 0.1)
        if isinstance(k, np.ndarray):
            k = k[0] if len(k) > 0 else 0.1

        zeta = self.param_values.get('zeta', 0.5)
        if isinstance(zeta, np.ndarray):
            zeta = zeta[0] if len(zeta) > 0 else 0.5

        omega = self.param_values.get('omega', 1.0)
        if isinstance(omega, np.ndarray):
            omega = omega[0] if len(omega) > 0 else 1.0

        # 时间范围
        if self.t_range:
            t = np.linspace(self.t_range[0], self.t_range[1], 100)
        else:
            t = np.linspace(0, 10, 100)

        # 根据方程类型求解
        if "d2y" in eq_str or "y''" in eq_str:
            # 二阶ODE（阻尼振动）
            def damped_vibration(y, t, zeta, omega):
                x, v = y
                dxdt = v
                dvdt = -2 * zeta * omega * v - omega ** 2 * x
                return [dxdt, dvdt]

            y0 = [1.0, 0.0]  # 初始条件
            solution = odeint(damped_vibration, y0, t, args=(zeta, omega))
            y_values = solution[:, 0]
        else:
            # 一阶ODE（衰减）
            def decay(y, t, k):
                return -k * y

            y0 = 1.0
            y_values = odeint(decay, y0, t, args=(k,)).flatten()

        return {
            'type': 'ode',
            't': t,
            'y': y_values,
            'equation': eq_str
        }

    def solve_pde(self):
        """求解偏微分方程（简化实现）"""
        eq_str = self.equation_data["equation"]

        # 获取参数
        alpha = self.param_values.get('alpha', 0.1)
        if isinstance(alpha, np.ndarray):
            alpha = alpha[0] if len(alpha) > 0 else 0.1

        c = self.param_values.get('c', 1.0)
        if isinstance(c, np.ndarray):
            c = c[0] if len(c) > 0 else 1.0

        D = self.param_values.get('D', 0.1)
        if isinstance(D, np.ndarray):
            D = D[0] if len(D) > 0 else 0.1

        r = self.param_values.get('r', 1.0)
        if isinstance(r, np.ndarray):
            r = r[0] if len(r) > 0 else 1.0

        K = self.param_values.get('K', 10.0)
        if isinstance(K, np.ndarray):
            K = K[0] if len(K) > 0 else 10.0

        # 空间和时间网格
        if self.x_range:
            x = np.linspace(self.x_range[0], self.x_range[1], 100)
        else:
            x = np.linspace(0, 10, 100)

        if self.t_range:
            t = np.linspace(self.t_range[0], self.t_range[1], 100)
        else:
            t = np.linspace(0, 5, 100)

        dx = x[1] - x[0]
        dt = t[1] - t[0]

        # 初始化
        u = np.zeros((len(t), len(x)))

        # 初始条件
        u[0, :] = np.exp(-(x - 5) ** 2)

        # 根据方程类型选择求解方法
        if "热传导" in eq_str or "alpha" in eq_str:
            # 热传导方程
            for n in range(len(t) - 1):
                for i in range(1, len(x) - 1):
                    u[n + 1, i] = u[n, i] + alpha * dt / dx ** 2 * (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1])
        elif "波动" in eq_str or "c**2" in eq_str:
            # 波动方程
            u[1, :] = u[0, :]  # 初始速度为0
            for n in range(1, len(t) - 1):
                for i in range(1, len(x) - 1):
                    u[n + 1, i] = 2 * u[n, i] - u[n - 1, i] + c ** 2 * dt ** 2 / dx ** 2 * (
                                u[n, i + 1] - 2 * u[n, i] + u[n, i - 1])
        else:
            # 反应扩散方程
            for n in range(len(t) - 1):
                for i in range(1, len(x) - 1):
                    diffusion = D * (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1]) / dx ** 2
                    reaction = r * u[n, i] * (1 - u[n, i] / K)
                    u[n + 1, i] = u[n, i] + dt * (diffusion + reaction)

        return {
            'type': 'pde',
            'x': x,
            't': t,
            'u': u,
            'equation': eq_str
        }


class ResultsWidget(QWidget):
    """结果展示组件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # 工具栏
        toolbar = QHBoxLayout()

        self.plot_btn = QPushButton("绘制图形")
        self.plot_btn.clicked.connect(self.plot_results)
        toolbar.addWidget(self.plot_btn)

        self.export_result_btn = QPushButton("导出结果")
        self.export_result_btn.clicked.connect(self.export_results)
        toolbar.addWidget(self.export_result_btn)

        self.clear_result_btn = QPushButton("清空结果")
        self.clear_result_btn.clicked.connect(self.clear_results)
        toolbar.addWidget(self.clear_result_btn)

        toolbar.addStretch()
        layout.addLayout(toolbar)

        # 分割器：文本结果和图形
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 文本结果
        text_widget = QWidget()
        text_layout = QVBoxLayout(text_widget)
        text_layout.addWidget(QLabel("计算结果:"))
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        text_layout.addWidget(self.result_text)
        splitter.addWidget(text_widget)

        # 图形区域
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.addWidget(QLabel("图形显示:"))

        self.canvas = MplCanvas(self, width=8, height=6)
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        splitter.addWidget(plot_widget)
        splitter.setSizes([300, 700])

        layout.addWidget(splitter)

        self.current_result = None

    def set_result(self, result, message=""):
        """设置计算结果"""
        self.current_result = result

        if result is None:
            self.result_text.append(f"错误: {message}")
            return

        # 显示文本结果
        info = [f"方程类型: {result['type']}", f"方程: {result['equation']}"]

        if result['type'] == 'algebraic':
            # 确保转换为 numpy 数组
            x_vals = np.array(result['x'], dtype=float)
            y_vals = np.array(result['y'], dtype=float)
            info.append(f"数据点数: {len(x_vals)}")
            info.append(f"X范围: [{x_vals.min():.4f}, {x_vals.max():.4f}]")
            info.append(f"Y范围: [{y_vals.min():.4f}, {y_vals.max():.4f}]")
        elif result['type'] == 'ode':
            t_vals = np.array(result['t'], dtype=float)
            y_vals = np.array(result['y'], dtype=float)
            info.append(f"时间点数: {len(t_vals)}")
            info.append(f"时间范围: [{t_vals.min():.4f}, {t_vals.max():.4f}]")
            info.append(f"Y范围: [{y_vals.min():.4f}, {y_vals.max():.4f}]")
        elif result['type'] == 'pde':
            u_vals = np.array(result['u'], dtype=float)
            x_vals = np.array(result['x'], dtype=float)
            t_vals = np.array(result['t'], dtype=float)
            info.append(f"空间点数: {len(x_vals)}")
            info.append(f"时间点数: {len(t_vals)}")
            info.append(f"U值范围: [{u_vals.min():.4f}, {u_vals.max():.4f}]")

        self.result_text.setHtml("<br>".join(info))

        # 自动绘制
        self.plot_results()

    def plot_results(self):
        """绘制结果图形"""
        if self.current_result is None:
            QMessageBox.information(self, "提示", "没有可绘制的数据")
            return

        result = self.current_result
        self.canvas.axes.clear()

        if result['type'] == 'algebraic':
            # 转换为 numpy 数组
            x_vals = np.array(result['x'], dtype=float)
            y_vals = np.array(result['y'], dtype=float)
            self.canvas.axes.plot(x_vals, y_vals, 'b-', linewidth=2)
            self.canvas.axes.set_xlabel('x')
            self.canvas.axes.set_ylabel('y')
            self.canvas.axes.set_title(f"代数方程: {result['equation']}")
            self.canvas.axes.grid(True)

        elif result['type'] == 'ode':
            t_vals = np.array(result['t'], dtype=float)
            y_vals = np.array(result['y'], dtype=float)
            self.canvas.axes.plot(t_vals, y_vals, 'r-', linewidth=2)
            self.canvas.axes.set_xlabel('t')
            self.canvas.axes.set_ylabel('y')
            self.canvas.axes.set_title(f"ODE: {result['equation']}")
            self.canvas.axes.grid(True)

        elif result['type'] == 'pde':
            # 转换为 numpy 数组
            x_vals = np.array(result['x'], dtype=float)
            t_vals = np.array(result['t'], dtype=float)
            u_vals = np.array(result['u'], dtype=float)

            X, T = np.meshgrid(x_vals, t_vals)

            # 使用子图
            self.canvas.fig.clear()

            # 3D表面图
            ax1 = self.canvas.fig.add_subplot(121, projection='3d')
            surf = ax1.plot_surface(X, T, u_vals, cmap='viridis')
            ax1.set_xlabel('x')
            ax1.set_ylabel('t')
            ax1.set_zlabel('u')
            ax1.set_title('3D视图')
            self.canvas.fig.colorbar(surf, ax=ax1, shrink=0.5)

            # 热力图
            ax2 = self.canvas.fig.add_subplot(122)
            im = ax2.imshow(u_vals, aspect='auto', origin='lower',
                            extent=[x_vals.min(), x_vals.max(),
                                    t_vals.min(), t_vals.max()],
                            cmap='hot')
            ax2.set_xlabel('x')
            ax2.set_ylabel('t')
            ax2.set_title('热力图')
            self.canvas.fig.colorbar(im, ax=ax2)

            self.canvas.fig.suptitle(f"PDE: {result['equation']}")

        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def export_results(self):
        """导出结果到文件"""
        if self.current_result is None:
            QMessageBox.information(self, "提示", "没有可导出的结果")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "", "NumPy文件 (*.npy);;CSV文件 (*.csv);;所有文件 (*.*)"
        )

        if file_path:
            try:
                result = self.current_result
                if file_path.endswith('.npy'):
                    np.save(file_path, result)
                else:
                    # 导出为CSV
                    if result['type'] in ['algebraic', 'ode']:
                        x_key = 'x' if result['type'] == 'algebraic' else 't'
                        df = pd.DataFrame({
                            x_key: result[x_key],
                            'y': result['y']
                        })
                        df.to_csv(file_path, index=False)
                    else:  # PDE
                        # 保存为多个CSV或reshape
                        df = pd.DataFrame(result['u'])
                        df.to_csv(file_path, index=False)

                QMessageBox.information(self, "成功", "结果导出成功")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")

    def clear_results(self):
        """清空结果"""
        self.current_result = None
        self.result_text.clear()
        self.canvas.axes.clear()
        self.canvas.draw()


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("数据分析与方程建模软件")
        self.setMinimumSize(1400, 900)

        self.setup_ui()
        self.setup_menu()
        self.setup_toolbar()
        self.setup_statusbar()

    def setup_ui(self):
        """设置主界面"""
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # 分割器
        splitter = QSplitter(Qt.Orientation.Vertical)

        # 上半部分：数据和方程
        top_splitter = QSplitter(Qt.Orientation.Horizontal)

        # 数据表格
        self.data_widget = DataTableWidget()
        self.data_widget.data_changed.connect(self.on_data_changed)
        top_splitter.addWidget(self.data_widget)

        # 方程管理
        self.equation_widget = EquationManagerWidget()
        self.equation_widget.equation_selected.connect(self.on_equation_selected)
        top_splitter.addWidget(self.equation_widget)

        # 参数关联
        self.param_widget = ParameterMappingWidget()
        top_splitter.addWidget(self.param_widget)

        top_splitter.setSizes([500, 400, 400])
        splitter.addWidget(top_splitter)

        # 下半部分：求解控制和结果
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)

        # 求解控制面板
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        # 变量范围设置
        range_group = QGroupBox("变量范围设置")
        range_layout = QFormLayout(range_group)

        self.x_min = QDoubleSpinBox()
        self.x_min.setRange(-1e6, 1e6)
        self.x_min.setValue(0)
        self.x_min.setDecimals(4)
        range_layout.addRow("X最小值:", self.x_min)

        self.x_max = QDoubleSpinBox()
        self.x_max.setRange(-1e6, 1e6)
        self.x_max.setValue(10)
        self.x_max.setDecimals(4)
        range_layout.addRow("X最大值:", self.x_max)

        self.t_min = QDoubleSpinBox()
        self.t_min.setRange(-1e6, 1e6)
        self.t_min.setValue(0)
        self.t_min.setDecimals(4)
        range_layout.addRow("T最小值:", self.t_min)

        self.t_max = QDoubleSpinBox()
        self.t_max.setRange(-1e6, 1e6)
        self.t_max.setValue(10)
        self.t_max.setDecimals(4)
        range_layout.addRow("T最大值:", self.t_max)

        control_layout.addWidget(range_group)

        # 求解按钮
        self.solve_btn = QPushButton("开始求解")
        self.solve_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 15px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.solve_btn.clicked.connect(self.start_solving)
        control_layout.addWidget(self.solve_btn)

        # 映射信息
        self.mapping_info = QLabel("参数映射: 未设置")
        self.mapping_info.setWordWrap(True)
        self.mapping_info.setStyleSheet("background-color: #f5f5f5; padding: 10px; border-radius: 5px;")
        control_layout.addWidget(self.mapping_info)

        control_layout.addStretch()
        bottom_splitter.addWidget(control_widget)

        # 结果展示
        self.results_widget = ResultsWidget()
        bottom_splitter.addWidget(self.results_widget)

        bottom_splitter.setSizes([300, 1100])
        splitter.addWidget(bottom_splitter)

        splitter.setSizes([400, 500])
        main_layout.addWidget(splitter)

        # 当前方程和数据
        self.current_equation = None
        self.solver_thread = None

    def setup_menu(self):
        """设置菜单栏"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件")

        open_action = QAction("打开CSV", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.data_widget.import_csv)
        file_menu.addAction(open_action)

        save_action = QAction("保存CSV", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.data_widget.export_csv)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction("退出", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 数据菜单
        data_menu = menubar.addMenu("数据")

        add_row_action = QAction("添加行", self)
        add_row_action.triggered.connect(self.data_widget.add_row)
        data_menu.addAction(add_row_action)

        add_col_action = QAction("添加列", self)
        add_col_action.triggered.connect(self.data_widget.add_column)
        data_menu.addAction(add_col_action)

        # 方程菜单
        eq_menu = menubar.addMenu("方程")

        add_eq_action = QAction("添加方程", self)
        add_eq_action.triggered.connect(self.equation_widget.add_equation)
        eq_menu.addAction(add_eq_action)

        # 帮助菜单
        help_menu = menubar.addMenu("帮助")

        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def setup_toolbar(self):
        """设置工具栏"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        toolbar.addAction("导入CSV", self.data_widget.import_csv)
        toolbar.addAction("导出CSV", self.data_widget.export_csv)
        toolbar.addSeparator()
        toolbar.addAction("添加方程", self.equation_widget.add_equation)
        toolbar.addSeparator()
        toolbar.addAction("开始求解", self.start_solving)

    def setup_statusbar(self):
        """设置状态栏"""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("就绪")

    def on_data_changed(self):
        """数据改变时更新"""
        columns = self.data_widget.get_column_names()
        self.param_widget.set_columns(columns)
        self.statusbar.showMessage(f"数据已更新，共 {len(columns)} 列")

    def on_equation_selected(self, name, data):
        """方程被选中时"""
        self.current_equation = data
        self.param_widget.set_parameters(data.get("params", []))
        self.update_mapping_info()
        self.statusbar.showMessage(f"已选择方程: {name}")

    def update_mapping_info(self):
        """更新映射信息显示"""
        mapping = self.param_widget.get_mapping_info()
        if mapping:
            self.mapping_info.setText("参数映射:\n" + "\n".join(mapping))
        else:
            self.mapping_info.setText("参数映射: 未设置")

    def start_solving(self):
        """开始求解"""
        if self.current_equation is None:
            QMessageBox.warning(self, "警告", "请先选择一个方程")
            return

        dataframe = self.data_widget.get_dataframe()
        x_range = (self.x_min.value(), self.x_max.value())
        t_range = (self.t_min.value(), self.t_max.value())

        param_values = self.param_widget.get_parameter_values(
            dataframe,
            x_values=np.linspace(x_range[0], x_range[1], 100),
            t_values=np.linspace(t_range[0], t_range[1], 100)
        )

        # 同步执行（测试用）
        try:
            self.solve_btn.setEnabled(False)
            self.solve_btn.setText("求解中...")

            solver = SolverThread(
                self.current_equation,
                param_values,
                dataframe,
                x_range,
                t_range
            )
            result = solver.solve()  # 直接调用，不用线程
            self.on_solver_finished(result, "计算完成")
        except Exception as e:
            self.on_solver_finished(None, f"错误: {str(e)}")
        finally:
            self.solve_btn.setEnabled(True)
            self.solve_btn.setText("开始求解")

    def on_solver_progress(self, message):
        """求解进度更新"""
        self.statusbar.showMessage(message)

    def on_solver_finished(self, result, message):
        """求解完成"""
        self.solve_btn.setEnabled(True)
        self.solve_btn.setText("开始求解")
        self.statusbar.showMessage(message)

        self.results_widget.set_result(result, message)

        if result is not None:
            QMessageBox.information(self, "完成", "计算完成！")

    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(self, "关于",
                          """<h2>数据分析与方程建模软件</h2>
                          <p>版本: 1.0</p>
                          <p>功能特点:</p>
                          <ul>
                              <li>支持CSV数据导入导出</li>
                              <li>手动数据编辑</li>
                              <li>丰富的预设方程库</li>
                              <li>自定义方程保存</li>
                              <li>参数与数据列灵活关联</li>
                              <li>支持代数方程、ODE、PDE求解</li>
                              <li>实时图形显示</li>
                          </ul>
                          <p>技术支持: Python + PyQt6 + NumPy + SciPy + SymPy + Matplotlib</p>
                          """)


def main():
    """主函数"""
    # 启用高DPI支持
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # 设置应用程序字体
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)

    # 设置样式表
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f5f5f5;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #cccccc;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            padding: 5px 15px;
            border: 1px solid #999999;
            border-radius: 3px;
            background-color: #e8e8e8;
        }
        QPushButton:hover {
            background-color: #d8d8d8;
        }
        QTableWidget {
            gridline-color: #cccccc;
        }
        QHeaderView::section {
            background-color: #e8e8e8;
            padding: 5px;
            border: 1px solid #cccccc;
        }
        QListWidget {
            border: 1px solid #cccccc;
            border-radius: 3px;
        }
        QListWidget::item {
            padding: 5px;
        }
        QListWidget::item:selected {
            background-color: #0078d7;
            color: white;
        }
        QLineEdit, QDoubleSpinBox, QComboBox {
            padding: 5px;
            border: 1px solid #cccccc;
            border-radius: 3px;
        }
        QTextEdit {
            border: 1px solid #cccccc;
            border-radius: 3px;
        }
    """)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()