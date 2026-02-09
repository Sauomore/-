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
9. 数据清洗功能
10. 多种图表类型支持
11. 纯净绘图模式
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
    QScrollArea, QCheckBox, QGridLayout, QInputDialog, QProgressBar,
    QRadioButton, QButtonGroup, QSizePolicy
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QAction, QFont, QIcon, QKeySequence

import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 版本号 - 在此处填写版本号
VERSION = "哈基米牌建模器 v1.2 beta"


def resource_path(relative_path):
    """获取资源绝对路径（兼容开发环境和打包后）"""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def to_latex_display(eq_str):
    """将方程转换为书面数学表达式显示"""
    display = eq_str
    # 替换幂运算
    display = re.sub(r'\*\*2', '²', display)
    display = re.sub(r'\*\*3', '³', display)
    display = re.sub(r'\*\*(\d+)', r'^\1', display)
    # 替换乘法
    display = display.replace('*', '·')
    # 替换除法
    display = display.replace('/', '÷')
    # 替换指数函数
    display = display.replace('exp(', 'e^(')
    # 替换平方根
    display = display.replace('sqrt(', '√(')
    # 替换希腊字母
    display = display.replace('alpha', 'α')
    display = display.replace('beta', 'β')
    display = display.replace('gamma', 'γ')
    display = display.replace('delta', 'δ')
    display = display.replace('sigma', 'σ')
    display = display.replace('mu', 'μ')
    display = display.replace('omega', 'ω')
    display = display.replace('zeta', 'ζ')
    display = display.replace('phi', 'φ')
    display = display.replace('pi', 'π')
    # 替换导数符号
    display = display.replace('dy/dt', 'dy/dt')
    display = display.replace('d2y/dt2', 'd²y/dt²')
    display = display.replace('du/dt', '∂u/∂t')
    display = display.replace('du/dx', '∂u/∂x')
    display = display.replace('d2u/dx2', '∂²u/∂x²')
    display = display.replace('d2u/dt2', '∂²u/∂t²')
    return display


def python_to_math_expr(expr):
    """将常用数学表达式转换为Python表达式"""
    result = expr
    # 替换幂运算
    result = result.replace('^', '**')
    result = result.replace('²', '**2')
    result = result.replace('³', '**3')
    # 替换乘法（中文点号）
    result = result.replace('·', '*')
    result = result.replace('×', '*')
    # 替换除法
    result = result.replace('÷', '/')
    # 替换指数函数
    result = re.sub(r'e\^\(([^)]+)\)', r'exp(\1)', result)
    # 替换平方根
    result = result.replace('√(', 'sqrt(')
    # 替换希腊字母
    result = result.replace('α', 'alpha')
    result = result.replace('β', 'beta')
    result = result.replace('γ', 'gamma')
    result = result.replace('δ', 'delta')
    result = result.replace('σ', 'sigma')
    result = result.replace('μ', 'mu')
    result = result.replace('ω', 'omega')
    result = result.replace('ζ', 'zeta')
    result = result.replace('φ', 'phi')
    result = result.replace('π', 'pi')
    return result


# 预设方程库 - 扩展更多常用方程
PRESET_EQUATIONS = {
    # 基础代数方程
    "线性回归": {
        "equation": "y = a * x + b",
        "display": "y = a·x + b",
        "params": ["a", "b"],
        "description": "一元线性回归方程"
    },
    "二次函数": {
        "equation": "y = a * x**2 + b * x + c",
        "display": "y = a·x² + b·x + c",
        "params": ["a", "b", "c"],
        "description": "二次多项式方程"
    },
    "三次函数": {
        "equation": "y = a * x**3 + b * x**2 + c * x + d",
        "display": "y = a·x³ + b·x² + c·x + d",
        "params": ["a", "b", "c", "d"],
        "description": "三次多项式方程"
    },
    "反比例函数": {
        "equation": "y = k / x",
        "display": "y = k ÷ x",
        "params": ["k"],
        "description": "反比例函数"
    },
    # 指数和对数
    "指数增长": {
        "equation": "y = a * exp(b * x)",
        "display": "y = a·e^(b·x)",
        "params": ["a", "b"],
        "description": "指数增长模型"
    },
    "指数衰减": {
        "equation": "y = a * exp(-b * x)",
        "display": "y = a·e^(-b·x)",
        "params": ["a", "b"],
        "description": "指数衰减模型"
    },
    "对数函数": {
        "equation": "y = a * log(x) + b",
        "display": "y = a·ln(x) + b",
        "params": ["a", "b"],
        "description": "自然对数函数"
    },
    "常用对数": {
        "equation": "y = a * log10(x) + b",
        "display": "y = a·log₁₀(x) + b",
        "params": ["a", "b"],
        "description": "常用对数函数"
    },
    "幂函数": {
        "equation": "y = a * x**b",
        "display": "y = a·x^b",
        "params": ["a", "b"],
        "description": "幂函数模型"
    },
    # S型和周期函数
    "逻辑斯蒂": {
        "equation": "y = L / (1 + exp(-k * (x - x0)))",
        "display": "y = L ÷ (1 + e^(-k·(x-x₀)))",
        "params": ["L", "k", "x0"],
        "description": "逻辑斯蒂增长模型（S型曲线）"
    },
    "双曲正切": {
        "equation": "y = a * tanh(b * x) + c",
        "display": "y = a·tanh(b·x) + c",
        "params": ["a", "b", "c"],
        "description": "双曲正切函数"
    },
    "正弦函数": {
        "equation": "y = A * sin(omega * x + phi) + C",
        "display": "y = A·sin(ω·x + φ) + C",
        "params": ["A", "omega", "phi", "C"],
        "description": "正弦波函数"
    },
    "余弦函数": {
        "equation": "y = A * cos(omega * x + phi) + C",
        "display": "y = A·cos(ω·x + φ) + C",
        "params": ["A", "omega", "phi", "C"],
        "description": "余弦波函数"
    },
    "阻尼振动": {
        "equation": "y = A * exp(-b * x) * sin(omega * x + phi)",
        "display": "y = A·e^(-b·x)·sin(ω·x + φ)",
        "params": ["A", "b", "omega", "phi"],
        "description": "阻尼振动函数"
    },
    # 概率分布
    "高斯分布": {
        "equation": "y = A * exp(-(x - mu)**2 / (2 * sigma**2))",
        "display": "y = A·e^(-(x-μ)²/(2σ²))",
        "params": ["A", "mu", "sigma"],
        "description": "高斯（正态）分布"
    },
    "瑞利分布": {
        "equation": "y = (x / sigma**2) * exp(-x**2 / (2 * sigma**2))",
        "display": "y = (x÷σ²)·e^(-x²/(2σ²))",
        "params": ["sigma"],
        "description": "瑞利分布"
    },
    # 常微分方程
    "一阶ODE-衰减": {
        "equation": "dy/dt = -k * y",
        "display": "dy/dt = -k·y",
        "params": ["k"],
        "description": "一阶线性常微分方程（衰减模型）"
    },
    "一阶ODE-增长": {
        "equation": "dy/dt = k * y",
        "display": "dy/dt = k·y",
        "params": ["k"],
        "description": "一阶线性常微分方程（增长模型）"
    },
    "逻辑斯蒂ODE": {
        "equation": "dy/dt = r * y * (1 - y / K)",
        "display": "dy/dt = r·y·(1 - y/K)",
        "params": ["r", "K"],
        "description": "逻辑斯蒂增长ODE"
    },
    "二阶ODE-阻尼": {
        "equation": "d2y/dt2 + 2*zeta*omega*dy/dt + omega**2*y = 0",
        "display": "d²y/dt² + 2ζω·dy/dt + ω²·y = 0",
        "params": ["zeta", "omega"],
        "description": "二阶线性常微分方程（阻尼振动）"
    },
    "简谐振动": {
        "equation": "d2y/dt2 + omega**2 * y = 0",
        "display": "d²y/dt² + ω²·y = 0",
        "params": ["omega"],
        "description": "简谐振动方程"
    },
    # 偏微分方程
    "热传导方程": {
        "equation": "du/dt = alpha * d2u/dx2",
        "display": "∂u/∂t = α·∂²u/∂x²",
        "params": ["alpha"],
        "description": "一维热传导方程"
    },
    "波动方程": {
        "equation": "d2u/dt2 = c**2 * d2u/dx2",
        "display": "∂²u/∂t² = c²·∂²u/∂x²",
        "params": ["c"],
        "description": "一维波动方程"
    },
    "反应扩散": {
        "equation": "du/dt = D * d2u/dx2 + r * u * (1 - u/K)",
        "display": "∂u/∂t = D·∂²u/∂x² + r·u·(1-u/K)",
        "params": ["D", "r", "K"],
        "description": "反应扩散方程（Fisher-KPP方程）"
    },
    "对流扩散": {
        "equation": "du/dt = D * d2u/dx2 - v * du/dx",
        "display": "∂u/∂t = D·∂²u/∂x² - v·∂u/∂x",
        "params": ["D", "v"],
        "description": "对流扩散方程"
    },
    # 经济金融模型
    "复利公式": {
        "equation": "y = P * (1 + r)**x",
        "display": "y = P·(1+r)^x",
        "params": ["P", "r"],
        "description": "复利计算公式"
    },
    "现值公式": {
        "equation": "y = FV / (1 + r)**x",
        "display": "y = FV ÷ (1+r)^x",
        "params": ["FV", "r"],
        "description": "现值计算公式"
    },
    "柯布道格拉斯": {
        "equation": "y = A * x1**alpha * x2**beta",
        "display": "y = A·x₁^α·x₂^β",
        "params": ["A", "alpha", "beta"],
        "description": "柯布-道格拉斯生产函数"
    }
}


class MplCanvas(FigureCanvas):
    """Matplotlib画布类"""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)


class PlotWindow(QDialog):
    """独立绘图窗口"""

    def __init__(self, parent=None, title="绘图窗口", width=10, height=8):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(800, 600)
        self.resize(1000, 800)

        layout = QVBoxLayout(self)

        # 创建画布
        self.fig = Figure(figsize=(width, height), dpi=100, tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.axes = self.fig.add_subplot(111)

    def clear(self):
        """清空图形"""
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)

    def draw(self):
        """绘制图形"""
        self.fig.tight_layout()
        self.canvas.draw()


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

        # 方程表达式（使用常用数学表达）
        self.eq_edit = QLineEdit(self.equation_data.get("equation", ""))
        self.eq_edit.setPlaceholderText("例如: y = a·x + b 或 y = a*x + b")
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

        # 帮助信息
        help_text = """
        <b>输入说明:</b><br>
        • 使用 <b>x</b> 作为自变量，<b>y</b> 作为因变量<br>
        • 可用符号: · (乘), ÷ (除), ^ 或 ** (幂), √ (根号), e^ (指数)<br>
        • 希腊字母: α, β, γ, δ, σ, μ, ω, ζ, φ, π<br>
        • 函数: sin, cos, tan, exp, log, log10, sqrt, tanh<br>
        """
        help_label = QLabel(help_text)
        help_label.setWordWrap(True)
        help_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        layout.addWidget(help_label)

        # 按钮
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_data(self):
        # 转换为Python表达式
        eq_str = python_to_math_expr(self.eq_edit.text())
        return {
            "name": self.name_edit.text(),
            "equation": eq_str,
            "display": to_latex_display(eq_str),
            "params": [p.strip() for p in self.params_edit.text().split(",") if p.strip()],
            "description": self.desc_edit.toPlainText()
        }


class ParamInputDialog(QDialog):
    """参数输入对话框（用于纯净绘图模式）"""

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.setWindowTitle("输入参数值")
        self.setMinimumWidth(300)
        self.params = params
        self.param_inputs = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        form_layout = QFormLayout()

        for param in self.params:
            spinbox = QDoubleSpinBox()
            spinbox.setRange(-1e10, 1e10)
            spinbox.setDecimals(6)
            spinbox.setValue(1.0)
            self.param_inputs[param] = spinbox
            form_layout.addRow(f"{param}:", spinbox)

        layout.addLayout(form_layout)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_param_values(self):
        return {param: spinbox.value() for param, spinbox in self.param_inputs.items()}


class DataTableWidget(QWidget):
    """数据表格组件 - 左侧数据区"""
    data_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataframe = None
        self.setup_ui()

    def setup_ui(self):
        """初始化数据表格组件的界面"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 数据操作工具栏
        toolbar = QHBoxLayout()

        self.import_btn = QPushButton("导入CSV")
        self.import_btn.setToolTip("从CSV文件导入数据")
        self.import_btn.clicked.connect(self.import_csv)
        toolbar.addWidget(self.import_btn)

        self.export_btn = QPushButton("导出CSV")
        self.export_btn.setToolTip("导出数据到CSV文件")
        self.export_btn.clicked.connect(self.export_csv)
        toolbar.addWidget(self.export_btn)

        toolbar.addSpacing(10)

        self.add_row_btn = QPushButton("+行")
        self.add_row_btn.clicked.connect(self.add_row)
        toolbar.addWidget(self.add_row_btn)

        self.add_col_btn = QPushButton("+列")
        self.add_col_btn.clicked.connect(self.add_column)
        toolbar.addWidget(self.add_col_btn)

        self.del_row_btn = QPushButton("-行")
        self.del_row_btn.clicked.connect(self.delete_row)
        toolbar.addWidget(self.del_row_btn)

        self.del_col_btn = QPushButton("-列")
        self.del_col_btn.clicked.connect(self.delete_column)
        toolbar.addWidget(self.del_col_btn)

        toolbar.addStretch()

        self.clear_btn = QPushButton("清空")
        self.clear_btn.clicked.connect(self.clear_data)
        toolbar.addWidget(self.clear_btn)

        main_layout.addLayout(toolbar)

        # 数据表格
        self.table = QTableWidget()
        self.table.setColumnCount(0)
        self.table.setRowCount(0)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.itemChanged.connect(self.on_item_changed)
        main_layout.addWidget(self.table)

        # 信息显示
        self.info_label = QLabel("暂无数据")
        main_layout.addWidget(self.info_label)

        self.setLayout(main_layout)

    def check_missing_values(self):
        """检测缺失值"""
        if self.dataframe is None or self.dataframe.empty:
            return "暂无数据"

        missing = self.dataframe.isnull().sum()
        missing_cols = missing[missing > 0]

        if len(missing_cols) == 0:
            return "✓ 未发现缺失值"
        else:
            result = "发现缺失值:\n"
            for col, count in missing_cols.items():
                result += f"  • {col}: {count} 个缺失值\n"
            return result

    def check_outliers(self):
        """检测异常值（使用IQR方法）"""
        if self.dataframe is None or self.dataframe.empty:
            return "暂无数据"

        result = "异常值检测结果 (IQR方法):\n"
        has_outliers = False

        for col in self.dataframe.select_dtypes(include=[np.number]).columns:
            data = self.dataframe[col].dropna()
            if len(data) < 4:
                continue

            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers = data[(data < lower) | (data > upper)]
            if len(outliers) > 0:
                has_outliers = True
                result += f"  • {col}: {len(outliers)} 个异常值\n"
                result += f"    范围: [{lower:.2f}, {upper:.2f}]\n"

        if not has_outliers:
            result += "✓ 未发现明显异常值"

        return result

    def fill_missing_values(self):
        """填充缺失值"""
        if self.dataframe is None or self.dataframe.empty:
            return False

        for col in self.dataframe.select_dtypes(include=[np.number]).columns:
            mean_val = self.dataframe[col].mean()
            self.dataframe[col].fillna(mean_val, inplace=True)

        self.update_table_from_df()
        self.data_changed.emit()
        return True

    def import_csv(self):
        """导入CSV文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择CSV文件", "", "CSV文件 (*.csv);;所有文件 (*.*)"
        )
        if file_path:
            try:
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
                    f"已加载: {os.path.basename(file_path)} | 行数: {len(self.dataframe)} | 列数: {len(self.dataframe.columns)}")
                self.data_changed.emit()
                QMessageBox.information(self, "成功", f"成功导入 {len(self.dataframe)} 行数据")
                return True
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导入失败: {str(e)}")
                return False
        return False

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


class PresetEquationsWidget(QWidget):
    """预设方程组件 - 左下角"""
    equation_selected = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 标题
        title_layout = QHBoxLayout()
        title_label = QLabel("【预设方程库】")
        title_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        layout.addLayout(title_layout)

        # 预设方程列表
        self.preset_list = QListWidget()
        self.preset_list.setMaximumHeight(200)  # 限制高度
        for name, data in PRESET_EQUATIONS.items():
            display_eq = data.get('display', data['equation'])
            item = QListWidgetItem(f"{name}: {display_eq}")
            item.setData(Qt.ItemDataRole.UserRole, {"name": name, **data})
            item.setToolTip(data.get("description", ""))
            self.preset_list.addItem(item)

        self.preset_list.itemDoubleClicked.connect(self.on_preset_selected)
        self.preset_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        layout.addWidget(self.preset_list)

        # 使用按钮
        self.use_preset_btn = QPushButton("使用选中方程")
        self.use_preset_btn.clicked.connect(self.use_selected_preset)
        layout.addWidget(self.use_preset_btn)

        # 当前选中方程显示
        self.current_eq_label = QLabel("未选择方程")
        self.current_eq_label.setWordWrap(True)
        self.current_eq_label.setStyleSheet(
            "font-size: 12px; color: #0066cc; padding: 5px; background-color: #f0f0f0; border-radius: 3px;")
        layout.addWidget(self.current_eq_label)

    def on_preset_selected(self, item):
        """预设方程被双击"""
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

    def set_current_equation(self, data):
        """设置当前方程"""
        display_eq = data.get('display', data['equation'])
        self.current_eq_label.setText(f"<b>{data['name']}</b><br>{display_eq}")
        self.equation_selected.emit(data["name"], data)


class DataCleaningWidget(QWidget):
    """数据清洗组件 - 压缩到右侧"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 标题
        title_layout = QHBoxLayout()
        title_label = QLabel("【数据清洗】")
        title_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        layout.addLayout(title_layout)

        # 检测按钮行
        btn_layout = QHBoxLayout()

        self.check_missing_btn = QPushButton("检测缺失值")
        self.check_missing_btn.setToolTip("检测数据中的缺失值")
        btn_layout.addWidget(self.check_missing_btn)

        self.check_outlier_btn = QPushButton("检测异常值")
        self.check_outlier_btn.setToolTip("使用IQR方法检测异常值")
        btn_layout.addWidget(self.check_outlier_btn)

        self.fill_missing_btn = QPushButton("填充缺失值")
        self.fill_missing_btn.setToolTip("使用列均值填充缺失值")
        btn_layout.addWidget(self.fill_missing_btn)

        layout.addLayout(btn_layout)

        # 结果显示 - 压缩高度
        self.anomaly_text = QTextEdit()
        self.anomaly_text.setMaximumHeight(80)
        self.anomaly_text.setPlaceholderText("数据检测结果将显示在这里...")
        self.anomaly_text.setReadOnly(True)
        layout.addWidget(self.anomaly_text)

    def set_check_missing_callback(self, callback):
        self.check_missing_btn.clicked.connect(callback)

    def set_check_outlier_callback(self, callback):
        self.check_outlier_btn.clicked.connect(callback)

    def set_fill_missing_callback(self, callback):
        self.fill_missing_btn.clicked.connect(callback)

    def set_result_text(self, text):
        self.anomaly_text.setText(text)


class EquationManagerWidget(QWidget):
    """方程管理组件 - 中间区域（简化版，只保留自定义方程和用户方程）"""
    equation_selected = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.user_equations = {}
        self.load_user_equations()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

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

        # 自定义方程输入区
        input_group = QGroupBox("自定义方程输入")
        input_layout = QVBoxLayout(input_group)

        self.custom_eq_input = QLineEdit()
        self.custom_eq_input.setPlaceholderText("输入方程，例如: y = a·x + b")
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

        # 用户方程页 - 简化显示
        user_group = QGroupBox("我的方程")
        user_layout = QVBoxLayout(user_group)

        self.user_list = QListWidget()
        self.refresh_user_list()
        self.user_list.itemDoubleClicked.connect(self.on_user_selected)
        user_layout.addWidget(self.user_list)

        user_btn_layout = QHBoxLayout()

        self.add_eq_btn = QPushButton("添加")
        self.add_eq_btn.clicked.connect(self.add_equation)
        user_btn_layout.addWidget(self.add_eq_btn)

        self.edit_eq_btn = QPushButton("编辑")
        self.edit_eq_btn.clicked.connect(self.edit_equation)
        user_btn_layout.addWidget(self.edit_eq_btn)

        self.del_eq_btn = QPushButton("删除")
        self.del_eq_btn.clicked.connect(self.delete_equation)
        user_btn_layout.addWidget(self.del_eq_btn)

        self.use_user_btn = QPushButton("使用")
        self.use_user_btn.clicked.connect(self.use_selected_user)
        user_btn_layout.addWidget(self.use_user_btn)

        user_layout.addLayout(user_btn_layout)
        layout.addWidget(user_group)

        # 纯净绘图模式
        pure_group = QGroupBox("纯净绘图模式")
        pure_layout = QVBoxLayout(pure_group)

        self.pure_eq_input = QLineEdit()
        self.pure_eq_input.setPlaceholderText("例如: y = x^2 或 y = sin(x)")
        pure_layout.addWidget(self.pure_eq_input)

        pure_btn_layout = QHBoxLayout()

        self.pure_plot_btn = QPushButton("绘制图形")
        self.pure_plot_btn.clicked.connect(self.pure_plot)
        pure_btn_layout.addWidget(self.pure_plot_btn)

        self.pure_range_btn = QPushButton("设置范围")
        self.pure_range_btn.clicked.connect(self.set_pure_range)
        pure_btn_layout.addWidget(self.pure_range_btn)

        pure_layout.addLayout(pure_btn_layout)

        # 纯净模式范围设置
        self.pure_x_min = -10
        self.pure_x_max = 10

        layout.addWidget(pure_group)

        # 帮助信息
        help_text = """
        <b>方程输入说明:</b><br>
        • <b>x</b> 自变量, <b>y</b> 因变量, <b>t</b> 时间<br>
        • 符号: · (乘), ÷ (除), ^ (幂), √ (根号)<br>
        • 希腊: α β γ δ σ μ ω ζ φ π<br>
        • ODE: dy/dt | PDE: ∂u/∂t, ∂²u/∂x²
        """
        help_label = QLabel(help_text)
        help_label.setWordWrap(True)
        help_label.setStyleSheet("background-color: #f0f0f0; padding: 8px; border-radius: 5px; font-size: 11px;")
        layout.addWidget(help_label)

        layout.addStretch()
        self.current_equation = None

    def set_pure_range(self):
        """设置纯净绘图的范围"""
        min_val, ok1 = QInputDialog.getDouble(self, "设置范围", "X最小值:", self.pure_x_min, -1e6, 1e6, 4)
        if ok1:
            max_val, ok2 = QInputDialog.getDouble(self, "设置范围", "X最大值:", self.pure_x_max, -1e6, 1e6, 4)
            if ok2 and max_val > min_val:
                self.pure_x_min = min_val
                self.pure_x_max = max_val

    def pure_plot(self):
        """纯净绘图模式"""
        eq_str = self.pure_eq_input.text().strip()
        if not eq_str:
            QMessageBox.warning(self, "警告", "请输入方程")
            return

        # 转换为Python表达式
        eq_str = python_to_math_expr(eq_str)

        # 提取参数
        params = self.extract_params(eq_str)

        # 如果有未知参数，弹出对话框
        param_values = {}
        if params:
            dialog = ParamInputDialog(params, self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            param_values = dialog.get_param_values()

        # 生成x值
        x_values = np.linspace(self.pure_x_min, self.pure_x_max, 500)

        try:
            # 解析方程
            if "=" in eq_str:
                left, right = eq_str.split("=", 1)
            else:
                right = eq_str

            # 创建计算环境
            safe_dict = {
                'x': x_values,
                'np': np,
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'exp': np.exp, 'log': np.log, 'log10': np.log10,
                'ln': np.log, 'sqrt': np.sqrt, 'abs': np.abs,
                'tanh': np.tanh,
                'pi': np.pi, 'e': np.e,
            }
            safe_dict.update(param_values)

            y_values = eval(right.strip(), {"__builtins__": {}}, safe_dict)
            y_values = np.array(y_values, dtype=float)

            if y_values.shape == ():
                y_values = np.full_like(x_values, y_values)

            # 创建绘图窗口
            plot_window = PlotWindow(self, title="纯净绘图模式", width=10, height=8)
            plot_window.axes.plot(x_values, y_values, 'b-', linewidth=2)
            plot_window.axes.set_xlabel('x')
            plot_window.axes.set_ylabel('y')
            plot_window.axes.set_title(f"y = {to_latex_display(right.strip())}")
            plot_window.axes.grid(True)
            plot_window.draw()
            plot_window.exec()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"绘图失败: {str(e)}")

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
            display_eq = data.get('display', data.get('equation', ''))
            item = QListWidgetItem(f"{name}: {display_eq}")
            item.setData(Qt.ItemDataRole.UserRole, {"name": name, **data})
            item.setToolTip(data.get("description", ""))
            self.user_list.addItem(item)

    def on_user_selected(self, item):
        """用户方程被双击"""
        data = item.data(Qt.ItemDataRole.UserRole)
        self.set_current_equation(data)

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
        display_eq = data.get('display', data['equation'])
        self.current_eq_label.setText(f"<b>{data['name']}</b><br>{display_eq}")
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
                    "display": data.get("display", to_latex_display(data["equation"])),
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
                if new_data["name"] != data["name"]:
                    del self.user_equations[data["name"]]
                self.user_equations[new_data["name"]] = {
                    "equation": new_data["equation"],
                    "display": new_data.get("display", to_latex_display(new_data["equation"])),
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

        # 转换为Python表达式
        eq_str = python_to_math_expr(eq_str)

        try:
            params = self.extract_params(eq_str)
            data = {
                "name": "自定义方程",
                "equation": eq_str,
                "display": to_latex_display(eq_str),
                "params": params,
                "description": "用户自定义方程"
            }
            self.set_current_equation(data)
            QMessageBox.information(self, "成功", f"方程解析成功！\n检测到参数: {', '.join(params) if params else '无'}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"方程解析失败: {str(e)}")

    def extract_params(self, eq_str):
        """从方程中提取参数"""
        if "=" in eq_str:
            _, right = eq_str.split("=", 1)
        else:
            right = eq_str

        known = {'x', 'y', 't', 'u', 'sin', 'cos', 'tan', 'exp', 'log', 'ln',
                 'sqrt', 'pi', 'e', 'abs', 'max', 'min', 'dy', 'dt', 'du', 'dx',
                 'log10', 'tanh', 'np'}

        words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', right)

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
                "display": self.current_equation.get("display", to_latex_display(self.current_equation["equation"])),
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
    """参数关联组件 - 中间区域（方程下方）"""
    mapping_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.column_names = []
        self.param_widgets = {}
        self.x_column = None  # x变量对应的列
        self.y_column = None  # y变量对应的列
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # X和Y含义指定区域
        xy_group = QGroupBox("变量含义指定")
        xy_layout = QFormLayout(xy_group)

        self.x_meaning_combo = QComboBox()
        self.x_meaning_combo.addItem("使用默认范围")
        self.x_meaning_combo.currentIndexChanged.connect(self.on_x_meaning_changed)
        xy_layout.addRow("X 代表:", self.x_meaning_combo)

        self.y_meaning_combo = QComboBox()
        self.y_meaning_combo.addItem("计算结果")
        self.y_meaning_combo.currentIndexChanged.connect(self.on_y_meaning_changed)
        xy_layout.addRow("Y 代表:", self.y_meaning_combo)

        layout.addWidget(xy_group)

        # 参数映射区域
        param_group = QGroupBox("参数与数据关联")
        param_layout = QVBoxLayout(param_group)

        info_label = QLabel("将方程参数与数据列关联")
        info_label.setStyleSheet("color: #666; font-style: italic;")
        param_layout.addWidget(info_label)

        # 参数映射区域（带滚动条）
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMaximumHeight(200)

        self.params_container = QWidget()
        self.params_layout = QFormLayout(self.params_container)
        self.params_layout.setSpacing(8)

        scroll.setWidget(self.params_container)
        param_layout.addWidget(scroll)

        # 快速设置按钮
        quick_layout = QHBoxLayout()

        self.clear_mapping_btn = QPushButton("清空映射")
        self.clear_mapping_btn.clicked.connect(self.clear_mapping)
        quick_layout.addWidget(self.clear_mapping_btn)

        self.auto_map_btn = QPushButton("自动映射")
        self.auto_map_btn.setToolTip("根据名称自动匹配参数和列")
        self.auto_map_btn.clicked.connect(self.auto_mapping)
        quick_layout.addWidget(self.auto_map_btn)

        quick_layout.addStretch()
        param_layout.addLayout(quick_layout)

        layout.addWidget(param_group)

        # 映射信息显示
        self.mapping_info = QLabel("参数映射: 未设置")
        self.mapping_info.setWordWrap(True)
        self.mapping_info.setStyleSheet("background-color: #f5f5f5; padding: 8px; border-radius: 3px;")
        layout.addWidget(self.mapping_info)

        layout.addStretch()

    def on_x_meaning_changed(self):
        """X含义改变"""
        self.x_column = self.x_meaning_combo.currentText()
        if self.x_column == "使用默认范围":
            self.x_column = None
        self.update_mapping_info()

    def on_y_meaning_changed(self):
        """Y含义改变"""
        self.y_column = self.y_meaning_combo.currentText()
        if self.y_column == "计算结果":
            self.y_column = None
        self.update_mapping_info()

    def set_columns(self, column_names):
        """设置可用的数据列"""
        self.column_names = column_names

        # 更新X和Y含义下拉框
        current_x = self.x_meaning_combo.currentText()
        current_y = self.y_meaning_combo.currentText()

        self.x_meaning_combo.clear()
        self.x_meaning_combo.addItem("使用默认范围")
        self.x_meaning_combo.addItems(self.column_names)

        self.y_meaning_combo.clear()
        self.y_meaning_combo.addItem("计算结果")
        self.y_meaning_combo.addItems(self.column_names)

        # 恢复之前的选择
        if current_x in self.column_names:
            self.x_meaning_combo.setCurrentText(current_x)
        if current_y in ["计算结果"] + self.column_names:
            self.y_meaning_combo.setCurrentText(current_y)

        # 更新参数控件中的列选择
        for param, widgets in self.param_widgets.items():
            type_combo = widgets["type_combo"]
            value_widget = widgets["value_widget"]

            if type_combo.currentIndex() == 0 and isinstance(value_widget, QComboBox):
                current_text = value_widget.currentText()
                value_widget.clear()
                value_widget.addItems(self.column_names)
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
            type_combo.addItems(["数据列", "常数值", "变量x", "变量t"])
            type_combo.setCurrentIndex(0)
            type_combo.currentIndexChanged.connect(lambda checked, p=param: self.on_type_changed(p))
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

        self.update_mapping_info()
        self.mapping_changed.emit()

    def on_type_changed(self, param):
        """参数类型改变时更新控件"""
        widgets = self.param_widgets[param]
        type_idx = widgets["type_combo"].currentIndex()

        container = widgets["type_combo"].parent()
        layout = container.layout()

        old_value_widget = widgets["value_widget"]
        layout.removeWidget(old_value_widget)
        old_value_widget.deleteLater()

        if type_idx == 0:  # 数据列
            new_widget = QComboBox()
            new_widget.addItems(self.column_names)
        elif type_idx == 1:  # 常数值
            new_widget = QDoubleSpinBox()
            new_widget.setRange(-1e10, 1e10)
            new_widget.setDecimals(6)
            new_widget.setValue(1.0)
        elif type_idx == 2:  # 变量x
            new_widget = QLabel("作为x变量")
            new_widget.setStyleSheet("color: gray;")
        else:  # 变量t
            new_widget = QLabel("作为t变量")
            new_widget.setStyleSheet("color: gray;")

        layout.addWidget(new_widget, 1)
        widgets["value_widget"] = new_widget

        container.update()
        self.update_mapping_info()
        self.mapping_changed.emit()

    def clear_mapping(self):
        """清空所有映射"""
        for param, widgets in self.param_widgets.items():
            widgets["type_combo"].setCurrentIndex(0)
            if isinstance(widgets["value_widget"], QComboBox):
                widgets["value_widget"].setCurrentIndex(0)
        self.update_mapping_info()

    def auto_mapping(self):
        """尝试自动映射参数到列"""
        for param, widgets in self.param_widgets.items():
            type_combo = widgets["type_combo"]
            value_widget = widgets["value_widget"]

            if param in self.column_names:
                type_combo.setCurrentIndex(0)
                if isinstance(value_widget, QComboBox):
                    value_widget.setCurrentText(param)
        self.update_mapping_info()

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
            elif type_idx == 1:  # 常数值
                if isinstance(value_widget, QDoubleSpinBox):
                    param_values[param] = value_widget.value()
                else:
                    param_values[param] = 0.0
            elif type_idx == 2:  # 变量x
                param_values[param] = x_values
            elif type_idx == 3:  # 变量t
                param_values[param] = t_values

        return param_values

    def get_x_column(self):
        """获取X对应的列名"""
        return self.x_column

    def get_y_column(self):
        """获取Y对应的列名"""
        return self.y_column

    def update_mapping_info(self):
        """更新映射信息显示"""
        info = []

        # X和Y含义
        x_desc = self.x_column if self.x_column else "使用默认范围"
        y_desc = self.y_column if self.y_column else "计算结果"
        info.append(f"X → {x_desc}")
        info.append(f"Y → {y_desc}")

        # 参数映射
        for param, widgets in self.param_widgets.items():
            type_idx = widgets["type_combo"].currentIndex()
            value_widget = widgets["value_widget"]

            type_names = ["数据列", "常数", "变量x", "变量t"]
            type_name = type_names[type_idx]

            if type_idx == 0 and isinstance(value_widget, QComboBox):
                value = value_widget.currentText()
            elif type_idx == 1 and isinstance(value_widget, QDoubleSpinBox):
                value = f"{value_widget.value():.4f}"
            else:
                value = type_name

            info.append(f"{param} → {value}")

        self.mapping_info.setText("\n".join(info))


class ResultsWidget(QWidget):
    """结果展示组件 - 右侧区域"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_result = None
        self.plot_window = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 控制面板
        control_group = QGroupBox("绘图设置")
        control_layout = QVBoxLayout(control_group)

        # 图表类型选择
        chart_type_layout = QHBoxLayout()
        chart_type_layout.addWidget(QLabel("图表类型:"))

        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems([
            "折线图", "散点图", "条形图", "水平条形图",
            "阶梯图", "填充图", "热力图", "3D表面图"
        ])
        chart_type_layout.addWidget(self.chart_type_combo)
        control_layout.addLayout(chart_type_layout)

        # 绘图窗口选择
        window_layout = QHBoxLayout()
        self.new_window_check = QCheckBox("在新窗口中绘图")
        self.new_window_check.setChecked(False)
        window_layout.addWidget(self.new_window_check)

        self.force_new_window_label = QLabel("(3D图将强制使用新窗口)")
        self.force_new_window_label.setStyleSheet("color: gray; font-size: 10px;")
        window_layout.addWidget(self.force_new_window_label)
        window_layout.addStretch()
        control_layout.addLayout(window_layout)

        # X轴和Y轴标签设置
        axis_layout = QFormLayout()
        self.x_label_input = QLineEdit("x")
        self.x_label_input.setPlaceholderText("X轴标签")
        axis_layout.addRow("X轴标签:", self.x_label_input)

        self.y_label_input = QLineEdit("y")
        self.y_label_input.setPlaceholderText("Y轴标签")
        axis_layout.addRow("Y轴标签:", self.y_label_input)

        control_layout.addLayout(axis_layout)

        # 绘图按钮
        btn_layout = QHBoxLayout()
        self.plot_btn = QPushButton("绘制图形")
        self.plot_btn.clicked.connect(self.plot_results)
        btn_layout.addWidget(self.plot_btn)

        self.plot_pure_btn = QPushButton("纯净绘图")
        self.plot_pure_btn.setToolTip("仅绘图，不依赖计算结果")
        self.plot_pure_btn.clicked.connect(self.pure_plot_dialog)
        btn_layout.addWidget(self.plot_pure_btn)

        control_layout.addLayout(btn_layout)

        layout.addWidget(control_group)

        # 运算控制区域
        calc_group = QGroupBox("运算控制")
        calc_layout = QVBoxLayout(calc_group)

        # 开始求解按钮
        self.solve_btn = QPushButton("开始求解")
        self.solve_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        calc_layout.addWidget(self.solve_btn)

        # 中断按钮
        self.stop_btn = QPushButton("中断运算")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.stop_btn.setEnabled(False)
        calc_layout.addWidget(self.stop_btn)

        # 进度条
        calc_layout.addWidget(QLabel("运算进度:"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        calc_layout.addWidget(self.progress_bar)

        layout.addWidget(calc_group)

        # 结果导出
        export_layout = QHBoxLayout()
        self.export_result_btn = QPushButton("导出结果")
        self.export_result_btn.clicked.connect(self.export_results)
        export_layout.addWidget(self.export_result_btn)

        self.clear_result_btn = QPushButton("清空结果")
        self.clear_result_btn.clicked.connect(self.clear_results)
        export_layout.addWidget(self.clear_result_btn)
        layout.addLayout(export_layout)

        # 文本结果显示
        result_group = QGroupBox("计算结果")
        result_layout = QVBoxLayout(result_group)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(150)
        result_layout.addWidget(self.result_text)
        layout.addWidget(result_group)

        # 图形区域
        plot_group = QGroupBox("图形显示")
        plot_layout = QVBoxLayout(plot_group)

        self.canvas = MplCanvas(self, width=6, height=4)
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        layout.addWidget(plot_group)

    def set_solve_callback(self, callback):
        """设置求解按钮回调"""
        self.solve_btn.clicked.connect(callback)

    def set_stop_callback(self, callback):
        """设置中断按钮回调"""
        self.stop_btn.clicked.connect(callback)

    def set_progress(self, value):
        """设置进度条值"""
        self.progress_bar.setValue(value)

    def set_solving_state(self, solving):
        """设置求解状态"""
        self.solve_btn.setEnabled(not solving)
        self.stop_btn.setEnabled(solving)
        if solving:
            self.solve_btn.setText("求解中...")
        else:
            self.solve_btn.setText("开始求解")
            self.progress_bar.setValue(0)

    def set_result(self, result, message=""):
        """设置计算结果"""
        self.current_result = result

        if result is None:
            self.result_text.setText(f"错误: {message}")
            return

        # 显示文本结果
        info = [f"方程类型: {result['type']}", f"方程: {result.get('display_eq', result['equation'])}"]

        if result['type'] == 'algebraic':
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

        self.result_text.setText("\n".join(info))

        # 自动绘制
        self.plot_results()

    def pure_plot_dialog(self):
        """纯净绘图对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("纯净绘图模式")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout(dialog)

        form_layout = QFormLayout()

        eq_input = QLineEdit()
        eq_input.setPlaceholderText("例如: y = x^2")
        form_layout.addRow("方程:", eq_input)

        x_min = QDoubleSpinBox()
        x_min.setRange(-1e6, 1e6)
        x_min.setValue(-10)
        x_min.setDecimals(2)
        form_layout.addRow("X最小值:", x_min)

        x_max = QDoubleSpinBox()
        x_max.setRange(-1e6, 1e6)
        x_max.setValue(10)
        x_max.setDecimals(2)
        form_layout.addRow("X最大值:", x_max)

        layout.addLayout(form_layout)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            eq_str = eq_input.text().strip()
            if eq_str:
                self.pure_plot(eq_str, x_min.value(), x_max.value())

    def pure_plot(self, eq_str, x_min, x_max):
        """纯净绘图"""
        eq_str = python_to_math_expr(eq_str)

        # 提取参数
        known = {'x', 'y', 't', 'sin', 'cos', 'tan', 'exp', 'log', 'ln',
                 'sqrt', 'pi', 'e', 'abs', 'log10', 'tanh', 'np'}
        words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', eq_str)
        params = [w for w in words if w not in known]

        # 如果有未知参数，弹出对话框
        param_values = {}
        if params:
            dialog = ParamInputDialog(params, self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            param_values = dialog.get_param_values()

        x_values = np.linspace(x_min, x_max, 500)

        try:
            if "=" in eq_str:
                left, right = eq_str.split("=", 1)
            else:
                right = eq_str

            safe_dict = {
                'x': x_values,
                'np': np,
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'exp': np.exp, 'log': np.log, 'log10': np.log10,
                'ln': np.log, 'sqrt': np.sqrt, 'abs': np.abs,
                'tanh': np.tanh,
                'pi': np.pi, 'e': np.e,
            }
            safe_dict.update(param_values)

            y_values = eval(right.strip(), {"__builtins__": {}}, safe_dict)
            y_values = np.array(y_values, dtype=float)

            if y_values.shape == ():
                y_values = np.full_like(x_values, y_values)

            # 创建绘图窗口
            plot_window = PlotWindow(self, title="纯净绘图", width=10, height=8)
            plot_window.axes.plot(x_values, y_values, 'b-', linewidth=2)
            plot_window.axes.set_xlabel(self.x_label_input.text() or 'x')
            plot_window.axes.set_ylabel(self.y_label_input.text() or 'y')
            plot_window.axes.set_title(f"y = {to_latex_display(right.strip())}")
            plot_window.axes.grid(True)
            plot_window.draw()
            plot_window.exec()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"绘图失败: {str(e)}")

    def plot_results(self):
        """绘制结果图形"""
        if self.current_result is None:
            QMessageBox.information(self, "提示", "没有可绘制的数据")
            return

        result = self.current_result
        chart_type = self.chart_type_combo.currentText()
        use_new_window = self.new_window_check.isChecked()

        # PDE强制使用新窗口
        if result['type'] == 'pde':
            use_new_window = True

        # 获取标签
        x_label = self.x_label_input.text() or 'x'
        y_label = self.y_label_input.text() or 'y'

        if use_new_window:
            self.plot_in_new_window(result, chart_type, x_label, y_label)
        else:
            self.plot_in_canvas(result, chart_type, x_label, y_label)

    def plot_in_canvas(self, result, chart_type, x_label, y_label):
        """在主画布中绘图"""
        self.canvas.axes.clear()

        try:
            if result['type'] == 'algebraic':
                x_vals = np.array(result['x'], dtype=float)
                y_vals = np.array(result['y'], dtype=float)

                if chart_type == "折线图":
                    self.canvas.axes.plot(x_vals, y_vals, 'b-', linewidth=2)
                elif chart_type == "散点图":
                    self.canvas.axes.scatter(x_vals, y_vals, c='blue', s=30)
                elif chart_type == "条形图":
                    self.canvas.axes.bar(x_vals, y_vals, width=0.8)
                elif chart_type == "阶梯图":
                    self.canvas.axes.step(x_vals, y_vals, where='mid')
                elif chart_type == "填充图":
                    self.canvas.axes.fill_between(x_vals, y_vals, alpha=0.5)
                else:
                    self.canvas.axes.plot(x_vals, y_vals, 'b-', linewidth=2)

                self.canvas.axes.set_xlabel(x_label)
                self.canvas.axes.set_ylabel(y_label)
                self.canvas.axes.set_title(f"{result.get('display_eq', result['equation'])}")
                self.canvas.axes.grid(True)

            elif result['type'] == 'ode':
                t_vals = np.array(result['t'], dtype=float)
                y_vals = np.array(result['y'], dtype=float)

                if chart_type == "折线图":
                    self.canvas.axes.plot(t_vals, y_vals, 'r-', linewidth=2)
                elif chart_type == "散点图":
                    self.canvas.axes.scatter(t_vals, y_vals, c='red', s=30)
                else:
                    self.canvas.axes.plot(t_vals, y_vals, 'r-', linewidth=2)

                self.canvas.axes.set_xlabel("时间 t")
                self.canvas.axes.set_ylabel(y_label)
                self.canvas.axes.set_title(f"ODE: {result.get('display_eq', result['equation'])}")
                self.canvas.axes.grid(True)

            self.canvas.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "绘图错误", f"绘图失败: {str(e)}")

    def plot_in_new_window(self, result, chart_type, x_label, y_label):
        """在新窗口中绘图"""
        try:
            plot_window = PlotWindow(self, title="绘图结果", width=12, height=8)

            if result['type'] == 'algebraic':
                x_vals = np.array(result['x'], dtype=float)
                y_vals = np.array(result['y'], dtype=float)

                if chart_type == "折线图":
                    plot_window.axes.plot(x_vals, y_vals, 'b-', linewidth=2)
                elif chart_type == "散点图":
                    plot_window.axes.scatter(x_vals, y_vals, c='blue', s=30)
                elif chart_type == "条形图":
                    plot_window.axes.bar(x_vals, y_vals, width=0.8)
                elif chart_type == "水平条形图":
                    plot_window.axes.barh(x_vals, y_vals, height=0.8)
                elif chart_type == "阶梯图":
                    plot_window.axes.step(x_vals, y_vals, where='mid')
                elif chart_type == "填充图":
                    plot_window.axes.fill_between(x_vals, y_vals, alpha=0.5)
                else:
                    plot_window.axes.plot(x_vals, y_vals, 'b-', linewidth=2)

                plot_window.axes.set_xlabel(x_label)
                plot_window.axes.set_ylabel(y_label)
                plot_window.axes.set_title(f"{result.get('display_eq', result['equation'])}")
                plot_window.axes.grid(True)

            elif result['type'] == 'ode':
                t_vals = np.array(result['t'], dtype=float)
                y_vals = np.array(result['y'], dtype=float)

                if chart_type == "散点图":
                    plot_window.axes.scatter(t_vals, y_vals, c='red', s=30)
                else:
                    plot_window.axes.plot(t_vals, y_vals, 'r-', linewidth=2)

                plot_window.axes.set_xlabel("时间 t")
                plot_window.axes.set_ylabel(y_label)
                plot_window.axes.set_title(f"ODE: {result.get('display_eq', result['equation'])}")
                plot_window.axes.grid(True)

            elif result['type'] == 'pde':
                x_vals = np.array(result['x'], dtype=float)
                t_vals = np.array(result['t'], dtype=float)
                u_vals = np.array(result['u'], dtype=float)

                X, T = np.meshgrid(x_vals, t_vals)

                plot_window.fig.clear()

                # 3D表面图
                ax1 = plot_window.fig.add_subplot(121, projection='3d')
                surf = ax1.plot_surface(X, T, u_vals, cmap='viridis')
                ax1.set_xlabel('空间 x')
                ax1.set_ylabel('时间 t')
                ax1.set_zlabel('u')
                ax1.set_title('3D视图')
                plot_window.fig.colorbar(surf, ax=ax1, shrink=0.5)

                # 热力图
                ax2 = plot_window.fig.add_subplot(122)
                im = ax2.imshow(u_vals, aspect='auto', origin='lower',
                                extent=[x_vals.min(), x_vals.max(),
                                        t_vals.min(), t_vals.max()],
                                cmap='hot')
                ax2.set_xlabel('空间 x')
                ax2.set_ylabel('时间 t')
                ax2.set_title('热力图')
                plot_window.fig.colorbar(im, ax=ax2)

                plot_window.fig.suptitle(f"PDE: {result.get('display_eq', result['equation'])}")

            plot_window.draw()
            plot_window.exec()

        except Exception as e:
            QMessageBox.critical(self, "绘图错误", f"绘图失败: {str(e)}")

    def export_results(self):
        """导出结果到文件"""
        if self.current_result is None:
            QMessageBox.information(self, "提示", "没有可导出的结果")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "", "CSV文件 (*.csv);;NumPy文件 (*.npy);;所有文件 (*.*)"
        )

        if file_path:
            try:
                result = self.current_result
                if file_path.endswith('.npy'):
                    np.save(file_path, result)
                else:
                    if result['type'] in ['algebraic', 'ode']:
                        x_key = 'x' if result['type'] == 'algebraic' else 't'
                        df = pd.DataFrame({
                            x_key: result[x_key],
                            'y': result['y']
                        })
                        df.to_csv(file_path, index=False)
                    else:
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
        self.progress_bar.setValue(0)


class MainWindow(QMainWindow):
    """主窗口 - 三段式布局"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("数据分析与方程建模软件")
        self.setMinimumSize(1600, 1000)
        self.resize(1800, 1100)

        self.current_equation = None
        self.is_solving = False
        self.stop_requested = False

        self.setup_ui()
        self.setup_menu()
        self.setup_toolbar()
        self.setup_statusbar()

    def setup_ui(self):
        """设置主界面 - 三段式布局"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # 水平分割器 - 三段式布局
        h_splitter = QSplitter(Qt.Orientation.Horizontal)

        # ========== 左侧：数据处理区 ==========
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_label = QLabel("【数据处理区】")
        left_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        left_layout.addWidget(left_label)

        # 数据表格组件
        self.data_widget = DataTableWidget()
        self.data_widget.data_changed.connect(self.on_data_changed)
        left_layout.addWidget(self.data_widget, 1)  # 给予更多空间

        # 数据清洗组件 - 压缩到右侧（放在左侧底部）
        self.cleaning_widget = DataCleaningWidget()
        self.cleaning_widget.setMaximumHeight(150)  # 限制高度
        self.cleaning_widget.set_check_missing_callback(self.on_check_missing)
        self.cleaning_widget.set_check_outlier_callback(self.on_check_outliers)
        self.cleaning_widget.set_fill_missing_callback(self.on_fill_missing)
        left_layout.addWidget(self.cleaning_widget)

        # 预设方程组件 - 移到左下角
        self.preset_widget = PresetEquationsWidget()
        self.preset_widget.equation_selected.connect(self.on_equation_selected)
        self.preset_widget.setMaximumHeight(280)  # 限制高度
        left_layout.addWidget(self.preset_widget)

        left_widget.setMinimumWidth(350)
        h_splitter.addWidget(left_widget)

        # ========== 中间：方程管理区 ==========
        middle_widget = QWidget()
        middle_layout = QVBoxLayout(middle_widget)
        middle_layout.setContentsMargins(0, 0, 0, 0)

        middle_label = QLabel("【方程管理区】")
        middle_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        middle_layout.addWidget(middle_label)

        # 方程管理组件
        self.equation_widget = EquationManagerWidget()
        self.equation_widget.equation_selected.connect(self.on_equation_selected)
        self.equation_widget.setMinimumHeight(400)
        self.equation_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        middle_layout.addWidget(self.equation_widget, 2)

        # 参数关联组件
        self.param_widget = ParameterMappingWidget()
        self.param_widget.mapping_changed.connect(self.on_mapping_changed)
        self.param_widget.setMinimumHeight(250)
        middle_layout.addWidget(self.param_widget, 1)

        # 变量范围设置
        range_group = QGroupBox("变量范围设置")
        range_group.setMaximumHeight(150)
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

        middle_layout.addWidget(range_group, 0)

        middle_widget.setMinimumWidth(380)
        h_splitter.addWidget(middle_widget)

        # ========== 右侧：输出结果区 ==========
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        right_label = QLabel("【输出结果区】")
        right_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        right_layout.addWidget(right_label)

        # 结果展示组件
        self.results_widget = ResultsWidget()
        self.results_widget.set_solve_callback(self.start_solving)
        self.results_widget.set_stop_callback(self.stop_solving)
        right_layout.addWidget(self.results_widget)

        right_widget.setMinimumWidth(400)
        h_splitter.addWidget(right_widget)

        # 设置分割器初始比例
        h_splitter.setSizes([450, 450, 500])

        main_layout.addWidget(h_splitter)

        # 底部版本号
        version_layout = QHBoxLayout()
        version_layout.addStretch()
        self.version_label = QLabel(f"版本: {VERSION}")
        self.version_label.setStyleSheet("color: gray; font-size: 10px;")
        version_layout.addWidget(self.version_label)
        main_layout.addLayout(version_layout)

    def on_check_missing(self):
        """检测缺失值"""
        result = self.data_widget.check_missing_values()
        self.cleaning_widget.set_result_text(result)

    def on_check_outliers(self):
        """检测异常值"""
        result = self.data_widget.check_outliers()
        self.cleaning_widget.set_result_text(result)

    def on_fill_missing(self):
        """填充缺失值"""
        success = self.data_widget.fill_missing_values()
        if success:
            self.cleaning_widget.set_result_text("✓ 已使用均值填充缺失值")

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

        check_missing_action = QAction("检测缺失值", self)
        check_missing_action.triggered.connect(self.on_check_missing)
        data_menu.addAction(check_missing_action)

        check_outlier_action = QAction("检测异常值", self)
        check_outlier_action.triggered.connect(self.on_check_outliers)
        data_menu.addAction(check_outlier_action)

        fill_missing_action = QAction("填充缺失值", self)
        fill_missing_action.triggered.connect(self.on_fill_missing)
        data_menu.addAction(fill_missing_action)

        # 方程菜单
        eq_menu = menubar.addMenu("方程")

        add_eq_action = QAction("添加方程", self)
        add_eq_action.triggered.connect(self.equation_widget.add_equation)
        eq_menu.addAction(add_eq_action)

        pure_plot_action = QAction("纯净绘图", self)
        pure_plot_action.triggered.connect(self.results_widget.pure_plot_dialog)
        eq_menu.addAction(pure_plot_action)

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
        toolbar.addAction("中断", self.stop_solving)

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
        self.statusbar.showMessage(f"已选择方程: {name}")

    def on_mapping_changed(self):
        """映射改变时"""
        pass

    def stop_solving(self):
        """中断求解"""
        self.stop_requested = True
        self.statusbar.showMessage("正在中断...")

    def start_solving(self):
        """开始求解 - 同步模式"""
        if self.current_equation is None:
            QMessageBox.warning(self, "警告", "请先选择一个方程")
            return

        self.is_solving = True
        self.stop_requested = False
        self.results_widget.set_solving_state(True)

        try:
            dataframe = self.data_widget.get_dataframe()
            x_range = (self.x_min.value(), self.x_max.value())
            t_range = (self.t_min.value(), self.t_max.value())

            # 获取X和Y含义
            x_column = self.param_widget.get_x_column()
            y_column = self.param_widget.get_y_column()

            # 生成x值
            if x_column and dataframe is not None and x_column in dataframe.columns:
                x_values = dataframe[x_column].values
            else:
                x_values = np.linspace(x_range[0], x_range[1], 100)

            t_values = np.linspace(t_range[0], t_range[1], 100)

            # 获取参数值
            param_values = self.param_widget.get_parameter_values(
                dataframe, x_values, t_values
            )

            # 更新进度
            self.results_widget.set_progress(10)
            self.statusbar.showMessage("正在解析方程...")
            QApplication.processEvents()

            if self.stop_requested:
                raise Exception("运算被用户中断")

            # 求解
            result = self.solve_equation(
                self.current_equation, param_values, x_values, t_values, x_column, y_column
            )

            self.results_widget.set_progress(100)
            self.statusbar.showMessage("计算完成")

            if result:
                self.results_widget.set_result(result, "计算完成")
                QMessageBox.information(self, "完成", "计算完成！")

        except Exception as e:
            self.results_widget.set_result(None, str(e))
            QMessageBox.critical(self, "错误", f"计算失败: {str(e)}")

        finally:
            self.is_solving = False
            self.results_widget.set_solving_state(False)

    def solve_equation(self, equation_data, param_values, x_values, t_values, x_column, y_column):
        """求解方程 - 同步模式"""
        eq_str = equation_data["equation"]
        display_eq = equation_data.get("display", to_latex_display(eq_str))

        # 检测方程类型
        if "d2u" in eq_str or "du/d" in eq_str:
            return self.solve_pde(eq_str, display_eq, param_values, x_values, t_values)
        elif "dy/d" in eq_str or "y'" in eq_str or "d2y" in eq_str:
            return self.solve_ode(eq_str, display_eq, param_values, t_values)
        else:
            return self.solve_algebraic(eq_str, display_eq, param_values, x_values, x_column, y_column)

    def solve_algebraic(self, eq_str, display_eq, param_values, x_values, x_column, y_column):
        """求解代数方程"""
        if self.stop_requested:
            raise Exception("运算被用户中断")

        self.results_widget.set_progress(30)
        self.statusbar.showMessage("正在计算代数方程...")
        QApplication.processEvents()

        # 解析方程
        if "=" in eq_str:
            left, right = eq_str.split("=", 1)
        else:
            right = eq_str

        # 准备参数
        param_dict = {}
        for param, value in param_values.items():
            if value is not None:
                if isinstance(value, np.ndarray):
                    param_dict[param] = float(value[0]) if len(value) > 0 else 0.0
                else:
                    param_dict[param] = float(value)
            else:
                param_dict[param] = 0.0

        x_values = np.array(x_values, dtype=float)

        safe_dict = {
            'x': x_values,
            'np': np,
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'exp': np.exp, 'log': np.log, 'log10': np.log10,
            'ln': np.log, 'sqrt': np.sqrt, 'abs': np.abs,
            'tanh': np.tanh,
            'pi': np.pi, 'e': np.e,
        }
        safe_dict.update(param_dict)

        expr_str = right.strip().replace('^', '**')
        y_values = eval(expr_str, {"__builtins__": {}}, safe_dict)
        y_values = np.array(y_values, dtype=float)

        if y_values.shape == ():
            y_values = np.full_like(x_values, y_values)

        self.results_widget.set_progress(80)

        return {
            'type': 'algebraic',
            'x': x_values,
            'y': y_values,
            'equation': eq_str,
            'display_eq': display_eq,
            'x_column': x_column,
            'y_column': y_column
        }

    def solve_ode(self, eq_str, display_eq, param_values, t_values):
        """求解常微分方程"""
        if self.stop_requested:
            raise Exception("运算被用户中断")

        self.results_widget.set_progress(30)
        self.statusbar.showMessage("正在求解ODE...")
        QApplication.processEvents()

        k = self.get_param_value(param_values, 'k', 0.1)
        zeta = self.get_param_value(param_values, 'zeta', 0.5)
        omega = self.get_param_value(param_values, 'omega', 1.0)
        r = self.get_param_value(param_values, 'r', 1.0)
        K = self.get_param_value(param_values, 'K', 10.0)

        t = np.array(t_values, dtype=float)

        self.results_widget.set_progress(50)

        if "d2y" in eq_str or "y''" in eq_str:
            def damped_vibration(y, t, zeta, omega):
                x, v = y
                dxdt = v
                dvdt = -2 * zeta * omega * v - omega ** 2 * x
                return [dxdt, dvdt]

            y0 = [1.0, 0.0]
            solution = odeint(damped_vibration, y0, t, args=(zeta, omega))
            y_values = solution[:, 0]
        elif "逻辑斯蒂" in eq_str or "r * y * (1" in eq_str:
            def logistic(y, t, r, K):
                return r * y * (1 - y / K)

            y0 = 0.1
            y_values = odeint(logistic, y0, t, args=(r, K)).flatten()
        else:
            def decay(y, t, k):
                return -k * y

            y0 = 1.0
            y_values = odeint(decay, y0, t, args=(k,)).flatten()

        self.results_widget.set_progress(80)

        return {
            'type': 'ode',
            't': t,
            'y': y_values,
            'equation': eq_str,
            'display_eq': display_eq
        }

    def solve_pde(self, eq_str, display_eq, param_values, x_values, t_values):
        """求解偏微分方程（隐式格式，无条件稳定）"""
        from scipy.sparse import diags
        from scipy.sparse.linalg import spsolve

        if self.stop_requested:
            raise Exception("运算被用户中断")

        self.results_widget.set_progress(20)
        self.statusbar.showMessage("正在求解PDE...")
        QApplication.processEvents()

        alpha = self.get_param_value(param_values, 'alpha', 0.1)
        c = self.get_param_value(param_values, 'c', 1.0)
        D = self.get_param_value(param_values, 'D', 0.1)
        r = self.get_param_value(param_values, 'r', 1.0)
        K = self.get_param_value(param_values, 'K', 10.0)
        v = self.get_param_value(param_values, 'v', 0.5)

        # 空间和时间网格（隐式格式可以用较大步长）
        x = np.array(x_values, dtype=float) if len(x_values) > 1 else np.linspace(0, 10, 100)
        t = np.array(t_values, dtype=float)

        dx = x[1] - x[0]
        dt = t[1] - t[0]

        N = len(x)
        u = np.zeros((len(t), N))
        u[0, :] = np.exp(-(x - np.mean(x)) ** 2 / 2)

        self.results_widget.set_progress(40)

        # 判断方程类型并求解
        if "热传导" in eq_str or "alpha" in eq_str.lower():
            # ========== 热传导方程（Crank-Nicolson隐式格式） ==========
            r_cn = alpha * dt / (2 * dx ** 2)

            # 构建三对角矩阵 A
            main_A = (1 + 2 * r_cn) * np.ones(N - 2)
            off_A = -r_cn * np.ones(N - 3)
            A = diags([off_A, main_A, off_A], [-1, 0, 1], format='csr')

            # 构建三对角矩阵 B
            main_B = (1 - 2 * r_cn) * np.ones(N - 2)
            off_B = r_cn * np.ones(N - 3)
            B = diags([off_B, main_B, off_B], [-1, 0, 1], format='csr')

            for n in range(len(t) - 1):
                if self.stop_requested:
                    raise Exception("运算被用户中断")
                if n % 10 == 0:
                    self.results_widget.set_progress(40 + int(40 * n / len(t)))
                    QApplication.processEvents()

                # 右端向量
                rhs = B.dot(u[n, 1:-1])
                rhs[0] += r_cn * u[n, 0]
                rhs[-1] += r_cn * u[n, -1]

                # 求解
                u[n + 1, 1:-1] = spsolve(A, rhs)
                u[n + 1, 0] = u[n + 1, 1]
                u[n + 1, -1] = u[n + 1, -2]

        elif "波动" in eq_str or "c**2" in eq_str:
            # ========== 波动方程 ==========
            u[1, :] = u[0, :]
            r_wave = c * dt / dx

            for n in range(1, len(t) - 1):
                if self.stop_requested:
                    raise Exception("运算被用户中断")
                if n % 10 == 0:
                    self.results_widget.set_progress(40 + int(40 * n / len(t)))
                    QApplication.processEvents()

                for i in range(1, N - 1):
                    u[n + 1, i] = 2 * u[n, i] - u[n - 1, i] + r_wave ** 2 * (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1])

                u[n + 1, 0] = u[n + 1, 1]
                u[n + 1, -1] = u[n + 1, -2]

        elif "对流" in eq_str or "-v" in eq_str:
            # ========== 对流扩散方程 ==========
            for n in range(len(t) - 1):
                if self.stop_requested:
                    raise Exception("运算被用户中断")

                for i in range(1, N - 1):
                    diffusion = D * (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1]) / dx ** 2
                    convection = -v * (u[n, i + 1] - u[n, i - 1]) / (2 * dx)
                    u[n + 1, i] = u[n, i] + dt * (diffusion + convection)

                u[n + 1, 0] = u[n + 1, 1]
                u[n + 1, -1] = u[n + 1, -2]
        else:
            # ========== 反应扩散方程 ==========
            r_diff = D * dt / dx ** 2

            main_A = (1 + 2 * r_diff) * np.ones(N - 2)
            off_A = -r_diff * np.ones(N - 3)
            A = diags([off_A, main_A, off_A], [-1, 0, 1], format='csr')

            for n in range(len(t) - 1):
                if self.stop_requested:
                    raise Exception("运算被用户中断")
                if n % 10 == 0:
                    self.results_widget.set_progress(40 + int(40 * n / len(t)))
                    QApplication.processEvents()

                reaction = r * u[n, 1:-1] * (1 - u[n, 1:-1] / K) * dt
                rhs = u[n, 1:-1] + r_diff * (u[n, 2:] - 2 * u[n, 1:-1] + u[n, :-2]) + reaction

                u[n + 1, 1:-1] = spsolve(A, rhs)
                u[n + 1, 0] = u[n + 1, 1]
                u[n + 1, -1] = u[n + 1, -2]

        # 检查结果
        if np.isnan(u).any() or np.isinf(u).any():
            print("警告：数值异常，已截断")
            u = np.nan_to_num(u, nan=0.0, posinf=1.0, neginf=0.0)

        print(f"U值范围: [{u.min():.4f}, {u.max():.4f}]")
        self.results_widget.set_progress(80)

        return {
            'type': 'pde',
            'x': x,
            't': t,
            'u': u,
            'equation': eq_str,
            'display_eq': display_eq
        }

    def get_param_value(self, param_values, param_name, default):
        """获取参数值"""
        value = param_values.get(param_name, default)
        if isinstance(value, np.ndarray):
            return float(value[0]) if len(value) > 0 else default
        return float(value) if value is not None else default

    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(self, "关于",
                          f"""<h2>数据分析与方程建模软件</h2>
                          <p>版本: {VERSION}</p>
                          <p>功能特点:</p>
                          <ul>
                              <li>支持CSV数据导入导出</li>
                              <li>数据清洗与异常检测</li>
                              <li>丰富的预设方程库</li>
                              <li>自定义方程保存</li>
                              <li>参数与数据列灵活关联</li>
                              <li>支持代数方程、ODE、PDE求解</li>
                              <li>多种图表类型</li>
                              <li>纯净绘图模式</li>
                          </ul>
                          <p>技术支持: Python + PyQt6 + NumPy + SciPy + Matplotlib</p>
                          """)


def main():
    """主函数"""
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)

    app.setStyleSheet("""
        QMainWindow {
            background-color: #f5f5f5;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #cccccc;
            border-radius: 5px;
            margin-top: 8px;
            padding-top: 8px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            padding: 5px 12px;
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
            padding: 4px;
            border: 1px solid #cccccc;
            border-radius: 3px;
        }
        QTextEdit {
            border: 1px solid #cccccc;
            border-radius: 3px;
        }
        QProgressBar {
            border: 1px solid #cccccc;
            border-radius: 3px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #4CAF50;
            border-radius: 3px;
        }
        QLabel {
            padding: 2px;
        }
    """)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()