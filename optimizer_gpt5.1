import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import threading
import time
import pandas as pd
from scipy.integrate import cumulative_trapezoid

# ==========================================
# 0. UX 增强工具: 悬浮提示 (Tooltip)
# ==========================================
class ToolTip(object):
    """
    为界面元素添加鼠标悬浮提示的专业工具类。
    """

    def __init__(self, widget, text='widget info'):
        self.wait_time = 500  # 毫秒
        self.wrap_length = 300  # 像素
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.wait_time, self.showtip)

    def unschedule(self):
        id_ = self.id
        self.id = None
        if id_:
            self.widget.after_cancel(id_)

    def showtip(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20

        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))

        label = tk.Label(
            self.tw,
            text=self.text,
            justify='left',
            background="#ffffe0",
            relief='solid',
            borderwidth=1,
            wraplength=self.wrap_length,
            font=("tahoma", "8", "normal")
        )
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tw
        self.tw = None
        if tw:
            tw.destroy()


# ==========================================
# 1. 物理模型 (Strip Mirror Model)
#    - 支持自适应增益校准 (Adaptive Gain Calibration)
# ==========================================
class StripDMModel:
    def __init__(self, positions, widths, amps):
        self.positions = np.array(positions)
        self.widths = np.array(widths)
        # Amps: 物理意义是 [Curvature / Voltage] (1/mm/V)
        # 初始值由用户输入，后续可通过 calibrate_gains 修正
        self.amps = np.array(amps, dtype=np.float64)
        self.n_actuators = len(self.positions)

        if len(self.widths) != self.n_actuators or len(self.amps) != self.n_actuators:
            raise ValueError(f"数据维度不匹配! Pos: {len(positions)}, Width: {len(widths)}, Amps: {len(amps)}")

    def influence_function(self, x_eval, act_idx, custom_amp=None):
        """
        计算单个致动器在 x_eval 处的曲率响应 (Unit Voltage)。
        """
        x0 = self.positions[act_idx]
        w = self.widths[act_idx]
        amp = custom_amp if custom_amp is not None else self.amps[act_idx]
        return amp * np.exp(-2.0 * ((x_eval - x0) ** 2) / (w ** 2))

    def get_active_curvature(self, x_eval, voltages):
        """
        计算由电压产生的“主动”变形量 (不包含初始面形)。
        """
        total_active_curve = np.zeros_like(x_eval)
        components = []

        for i in range(self.n_actuators):
            comp = voltages[i] * self.influence_function(x_eval, i)
            total_active_curve += comp
            components.append(comp)

        return total_active_curve, components

    def calibrate_gains(self, voltages_applied, measured_delta_curvature, x_coords,
                        threshold_v=3.0, learn_rate=0.6):
        """
        自适应校准核心算法。
        根据实测反馈修正 self.amps。
        """
        predicted_delta, _ = self.get_active_curvature(x_coords, voltages_applied)
        new_amps = self.amps.copy()
        log_info = []

        for i in range(self.n_actuators):
            v = voltages_applied[i]
            # 只有当施加了足够大的电压时，才进行校准
            if abs(v) < threshold_v:
                continue

            center_x = self.positions[i]
            meas_at_center = np.interp(center_x, x_coords, measured_delta_curvature)
            pred_at_center = np.interp(center_x, x_coords, predicted_delta)

            if abs(pred_at_center) < 1e-15:
                continue

            # Ratio = Real / Model
            ratio = meas_at_center / pred_at_center

            # 安全限制：防止单次修改过大
            ratio = np.clip(ratio, 0.2, 2.5)

            # 平滑更新
            update_factor = (1.0 - learn_rate) + learn_rate * ratio

            old_amp = self.amps[i]
            new_amp = old_amp * update_factor
            new_amps[i] = new_amp

            log_info.append(
                f"Act {i + 1}: V={v:.1f}, Ratio={ratio:.2f} -> Amp: {old_amp:.2e} to {new_amp:.2e}"
            )

        self.amps = new_amps
        return log_info


# ==========================================
# 2. PSO 优化器 (支持 Baseline & 历史记录)
# ==========================================
class PSOOptimizer1D:
    def __init__(self, dm_model, target_x, target_y, baseline_y=None,
                 voltage_limits=50.0, n_particles=50, initial_guess=None):
        self.dm = dm_model
        self.target_x = target_x
        self.target_y = target_y

        # 处理基准面形
        if baseline_y is not None:
            self.baseline_y = baseline_y
        else:
            self.baseline_y = np.zeros_like(target_y)

        # 目标：Active(V) = Target - Baseline
        self.effective_target_y = self.target_y - self.baseline_y

        self.n_particles = n_particles
        self.dim = dm_model.n_actuators

        if isinstance(voltage_limits, (int, float)):
            self.voltage_limits = np.full(self.dim, float(voltage_limits))
        else:
            self.voltage_limits = np.array(voltage_limits)

        # 预计算影响矩阵
        self.influence_matrix = np.zeros((len(target_x), self.dim))
        for i in range(self.dim):
            self.influence_matrix[:, i] = self.dm.influence_function(target_x, i)

        # 初始化粒子
        self.limit_scalar = np.max(np.abs(self.voltage_limits))
        self.positions = np.random.uniform(
            -self.limit_scalar / 5, self.limit_scalar / 5, (n_particles, self.dim)
        )

        if initial_guess is not None:
            if len(initial_guess) == self.dim:
                self.positions[0] = initial_guess.copy()
                # 在猜测值附近撒点
                for k in range(1, min(10, n_particles)):
                    self.positions[k] = initial_guess + np.random.normal(0, 1.0, self.dim)

        self.velocities = np.random.uniform(-1, 1, (n_particles, self.dim))

        self.pbest_pos = self.positions.copy()
        self.pbest_score = np.full(n_particles, np.inf)
        self.gbest_pos = np.zeros(self.dim)
        self.gbest_score = np.inf

        # 记录每一次迭代的最佳电压向量，用于绘制热力图
        self.gbest_history = []

    def calculate_merit(self, voltage_set):
        generated_active_y = self.influence_matrix @ voltage_set
        error = generated_active_y - self.effective_target_y
        return np.mean(error ** 2)

    def step(self, w=0.7, c1=1.49, c2=1.49):
        # 1. 更新速度与位置
        r1 = np.random.rand(self.n_particles, self.dim)
        r2 = np.random.rand(self.n_particles, self.dim)

        self.velocities = (w * self.velocities +
                           c1 * r1 * (self.pbest_pos - self.positions) +
                           c2 * r2 * (self.gbest_pos - self.positions))

        self.positions += self.velocities

        # 边界限制
        for j in range(self.dim):
            self.positions[:, j] = np.clip(
                self.positions[:, j],
                -self.voltage_limits[j],
                self.voltage_limits[j]
            )

        # 2. 评估
        all_generated = self.influence_matrix @ self.positions.T
        err_matrix = all_generated - self.effective_target_y[:, np.newaxis]
        current_scores = np.mean(err_matrix ** 2, axis=0)

        # 3. 更新 Best
        improved_idx = current_scores < self.pbest_score
        self.pbest_pos[improved_idx] = self.positions[improved_idx]
        self.pbest_score[improved_idx] = current_scores[improved_idx]

        min_idx = np.argmin(current_scores)
        if current_scores[min_idx] < self.gbest_score:
            self.gbest_score = current_scores[min_idx]
            self.gbest_pos = self.positions[min_idx].copy()

        # 保存历史
        self.gbest_history.append(self.gbest_pos.copy())

        return self.gbest_score, self.gbest_pos


# ==========================================
# 3. 新增模块：线性解、电压安全、目标管理、优化器工厂、日志、自动保存
# ==========================================

class LinearLeastSquaresSolver:
    """
    使用线性最小二乘法求解电压： A V ≈ Target
    """
    def __init__(self, dm_model, target_x, target_y, baseline_y=None, voltage_limits=50.0):
        self.dm = dm_model
        self.target_x = np.array(target_x)
        self.target_y = np.array(target_y)

        if baseline_y is not None:
            self.baseline_y = np.array(baseline_y)
        else:
            self.baseline_y = np.zeros_like(self.target_y)

        self.effective_target_y = self.target_y - self.baseline_y
        self.dim = dm_model.n_actuators

        if isinstance(voltage_limits, (int, float)):
            self.voltage_limits = np.full(self.dim, float(voltage_limits))
        else:
            self.voltage_limits = np.array(voltage_limits)

        # 预计算影响矩阵
        self.influence_matrix = np.zeros((len(self.target_x), self.dim))
        for i in range(self.dim):
            self.influence_matrix[:, i] = self.dm.influence_function(self.target_x, i)

    def solve(self):
        A = self.influence_matrix
        b = self.effective_target_y
        # 最小二乘
        v, *_ = np.linalg.lstsq(A, b, rcond=None)
        # 限幅
        v = np.clip(v, -self.voltage_limits, self.voltage_limits)
        # merit
        err = A @ v - b
        merit = np.mean(err ** 2)
        return v, merit


class VoltageSafetyChecker:
    """
    电压安全检查：超限 & ΔV 限制
    """
    def __init__(self, limit_value=50.0, max_delta_per_step=10.0):
        self.limit_value = float(limit_value)
        self.max_delta_per_step = float(max_delta_per_step)

    def check_limits(self, voltages):
        v = np.array(voltages)
        over = np.any(np.abs(v) > self.limit_value + 1e-9)
        if over:
            idx = np.argmax(np.abs(v))
            return False, f"电压超限: Act {idx + 1}, V={v[idx]:.2f} V (Limit={self.limit_value:.2f} V)"
        return True, "电压在安全范围内"

    def check_delta(self, new_voltages, last_voltages):
        if last_voltages is None:
            return True, "无历史电压，跳过 ΔV 检查"
        new = np.array(new_voltages)
        old = np.array(last_voltages)
        if new.shape != old.shape:
            return True, "电压维度变化，跳过 ΔV 检查"

        delta = new - old
        max_delta = np.max(np.abs(delta))
        if max_delta > self.max_delta_per_step:
            idx = np.argmax(np.abs(delta))
            return False, f"电压变化过大: Act {idx + 1}, ΔV={delta[idx]:.2f} V (Limit={self.max_delta_per_step:.2f} V)"
        return True, "ΔV 在安全范围内"


class TargetCurveManager:
    """
    目标曲线管理：生成/导入
    """
    def __init__(self):
        self.current_x = None
        self.current_y = None
        self.mode = None
        self.metadata = {}

    def generate(self, positions, mode="flat", extra=None):
        pos = np.array(positions)
        x = np.linspace(np.min(pos) - 5, np.max(pos) + 5, 400)
        if mode == "flat":
            y = np.zeros_like(x)
        elif mode == "parabola":
            center = np.mean(pos)
            scale = 1e-9
            if extra is not None and "scale" in extra:
                scale = float(extra["scale"])
            y = scale * ((x - center) ** 2) / 1000.0
        elif mode == "sine":
            # 简单正弦曲面
            period = extra.get("period", (np.max(pos) - np.min(pos))) if extra else (np.max(pos) - np.min(pos))
            amp = extra.get("amp", 5e-9) if extra else 5e-9
            center = np.mean(pos)
            y = amp * np.sin(2 * np.pi * (x - center) / period)
        else:
            raise ValueError(f"未知目标模式: {mode}")

        self.current_x = x
        self.current_y = y
        self.mode = mode
        self.metadata = extra or {}
        return x, y

    def load_from_file(self, path):
        df = pd.read_csv(path, header=None, sep=None, engine='python')
        data = df.values
        if data.shape[1] < 2:
            raise ValueError("目标文件需要至少两列数据 (x, y)")
        data = data[data[:, 0].argsort()]
        self.current_x, self.current_y = data[:, 0], data[:, 1]
        self.mode = "file"
        self.metadata = {"path": path}
        return self.current_x, self.current_y


class LogManager:
    """
    简易日志系统，后续可以替换为标准 logging 或文件日志
    """
    def __init__(self):
        self.logs = []

    def add(self, msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.logs.append(line)
        return line

    def get_all(self):
        return "\n".join(self.logs)


class AutoSaveManager:
    """
    自动保存优化结果和简单配置
    """
    def __init__(self):
        self.last_save_dir = None

    def auto_save_results(self, voltages, meta=None):
        if voltages is None:
            return
        # 简单策略：询问一次目录，然后固定到该目录
        if self.last_save_dir is None:
            d = filedialog.askdirectory(title="选择结果保存目录 (AutoSave)")
            if not d:
                return
            self.last_save_dir = d

        t_str = time.strftime("%Y%m%d_%H%M%S")
        base = self.last_save_dir.rstrip("/\\")
        v_path = f"{base}/voltages_{t_str}.csv"
        pd.DataFrame({"Voltage": voltages}).to_csv(v_path, index=False)

        if meta is not None:
            try:
                import json
                m_path = f"{base}/meta_{t_str}.json"
                with open(m_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            except Exception:
                pass


class OptimizerFactory:
    """
    优化器工厂：根据用户选择返回合适的优化流程
    当前支持:
      - "PSO"
      - "Linear Only" (线性最小二乘)
      - "Linear + PSO" (先线性解作为初值，再 PSO 微调)
    """
    MODE_PSO = "PSO"
    MODE_LINEAR = "Linear Only"
    MODE_HYBRID = "Linear + PSO"

    def __init__(self):
        pass

    def create(self, mode, dm_model, target_x, target_y,
               baseline_y, voltage_limit, n_iter, n_particles=40):
        """
        返回 (optimizer, linear_solver, run_mode)
          - optimizer: PSOOptimizer1D 或 None
          - linear_solver: LinearLeastSquaresSolver 或 None
          - run_mode: 字符串用于上层逻辑判断
        """
        mode = mode or self.MODE_PSO

        if mode == self.MODE_LINEAR:
            lin = LinearLeastSquaresSolver(dm_model, target_x, target_y,
                                           baseline_y=baseline_y,
                                           voltage_limits=voltage_limit)
            return None, lin, self.MODE_LINEAR

        if mode == self.MODE_HYBRID:
            lin = LinearLeastSquaresSolver(dm_model, target_x, target_y,
                                           baseline_y=baseline_y,
                                           voltage_limits=voltage_limit)
            opt = PSOOptimizer1D(dm_model, target_x, target_y,
                                 baseline_y=baseline_y,
                                 voltage_limits=voltage_limit,
                                 n_particles=n_particles,
                                 initial_guess=None)  # 初值稍后设置
            return opt, lin, self.MODE_HYBRID

        # 默认: PSO
        opt = PSOOptimizer1D(dm_model, target_x, target_y,
                             baseline_y=baseline_y,
                             voltage_limits=voltage_limit,
                             n_particles=n_particles)
        return opt, None, self.MODE_PSO


# ==========================================
# 4. GUI 界面 (Pro版: 集成RMS/PV显示与热力图 + 新模块)
# ==========================================

class StripDMApp:
    def __init__(self, root):
        self.root = root
        self.root.title("1D Strip DM Optimizer (Pro Edition+)")
        self.root.geometry("1550x900")

        # --- 核心状态变量 ---
        self.dm_model = None
        self.optimizer = None
        self.is_running = False
        self.history = []

        # 流程控制变量
        self.baseline_x = None
        self.baseline_y = None  # 初始状态 (电压=0 或 上一时刻状态)
        self.last_optimized_voltages = None
        self.last_target_x = None

        # 新增模块实例
        self.target_manager = TargetCurveManager()
        self.log_manager = LogManager()
        self.auto_saver = AutoSaveManager()
        self.safety_checker = None
        self.optimizer_factory = OptimizerFactory()
        self.current_optimizer_mode = OptimizerFactory.MODE_PSO
        self.linear_solver = None  # 用于预览或 hybrid

        # GUI 组件引用
        self.combo_optimizer = None
        self.txt_log = None

        # 快捷键绑定
        self._bind_shortcuts()

        self._setup_ui()
        self._setup_plots()
        self.fill_defaults()

    def _bind_shortcuts(self):
        """绑定快捷键，提升操作效率"""
        self.root.bind('<Control-Return>', lambda event: self.start_opt())
        self.root.bind('<Control-o>', lambda event: self.load_baseline_from_file())
        self.root.bind('<Control-s>', lambda event: self.save_results())

    def _setup_ui(self):
        # 左右分栏
        container = tk.Frame(self.root)
        container.pack(side=tk.LEFT, fill=tk.Y)

        canvas = tk.Canvas(container, width=520, height=900)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        self.left_panel = tk.Frame(canvas, padx=5, pady=5)
        canvas.create_window((0, 0), window=self.left_panel, anchor="nw")

        self.left_panel.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ================== 1. 硬件参数 ==================
        frame_hw = ttk.LabelFrame(self.left_panel, text="1. 硬件参数 (Physics)", padding="5")
        frame_hw.pack(fill=tk.X, pady=5)

        self._add_input_row(frame_hw, "位置 (mm):", "txt_pos")
        self._add_input_row(frame_hw, "宽度 (mm):", "txt_width")
        self._add_input_row(frame_hw, "幅度 Amps (输入值):", "txt_amp")

        btn_reset = ttk.Button(frame_hw, text="重置为默认参数", command=self.fill_defaults)
        btn_reset.pack(pady=2)

        # ================== 2. 闭环工作流 ==================
        frame_flow = ttk.LabelFrame(self.left_panel, text="2. 闭环校正工作流", padding="5")
        frame_flow.pack(fill=tk.X, pady=5)

        # --- Step A: Baseline ---
        step_a = ttk.LabelFrame(frame_flow, text="Step A: 确立基准 (Ctrl+O)", padding=2)
        step_a.pack(fill=tk.X, pady=2)

        self.lbl_baseline_status = ttk.Label(step_a, text="状态: 未加载 (默认0)", foreground="red")
        self.lbl_baseline_status.pack(side=tk.TOP, anchor="w")

        btn_base = ttk.Button(step_a, text="导入当前实测面形 (Baseline)", command=self.load_baseline_from_file)
        btn_base.pack(fill=tk.X, pady=2)
        ToolTip(
            btn_base,
            "导入电压为0时(或当前状态)的曲率数据。\n解决'电压为0面形不为0'的问题。"
        )

        ttk.Button(step_a, text="清除基准", command=self.clear_baseline).pack(fill=tk.X)

        # --- Step B: 设定目标 & 优化 ---
        step_b = ttk.LabelFrame(frame_flow, text="Step B: 设定目标 & 优化 (Ctrl+Enter)", padding=2)
        step_b.pack(fill=tk.X, pady=2)

        # 目标选择行
        btn_tgt_frame = ttk.Frame(step_b)
        btn_tgt_frame.pack(fill=tk.X)
        ttk.Button(
            btn_tgt_frame,
            text="目标: 平面 (Flat)",
            command=lambda: self.gen_target("flat")
        ).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(
            btn_tgt_frame,
            text="目标: 抛物面",
            command=lambda: self.gen_target("parabola")
        ).pack(side=tk.LEFT, expand=True, fill=tk.X)

        ttk.Button(
            step_b,
            text="目标: 正弦 (Sine)",
            command=lambda: self.gen_target("sine")
        ).pack(fill=tk.X, pady=2)

        btn_tgt_load = ttk.Button(step_b, text="从文件导入目标曲线", command=self.load_target_from_file)
        btn_tgt_load.pack(fill=tk.X, pady=2)
        ToolTip(btn_tgt_load, "从 CSV/TXT 导入 (x, y) 作为目标面形。")

        # 优化参数行
        param_frame = ttk.Frame(step_b)
        param_frame.pack(fill=tk.X, pady=2)
        ttk.Label(param_frame, text="Iter:").pack(side=tk.LEFT)
        self.entry_iter = ttk.Entry(param_frame, width=6)
        self.entry_iter.insert(0, "200")
        self.entry_iter.pack(side=tk.LEFT, padx=5)
        ttk.Label(param_frame, text="Limit(V):").pack(side=tk.LEFT)
        self.entry_limit = ttk.Entry(param_frame, width=6)
        self.entry_limit.insert(0, "50")
        self.entry_limit.pack(side=tk.LEFT)

        ttk.Label(param_frame, text="ΔVmax:").pack(side=tk.LEFT)
        self.entry_dvmax = ttk.Entry(param_frame, width=6)
        self.entry_dvmax.insert(0, "10")
        self.entry_dvmax.pack(side=tk.LEFT)

        # 优化器选择
        opt_frame = ttk.Frame(step_b)
        opt_frame.pack(fill=tk.X, pady=2)
        ttk.Label(opt_frame, text="优化模式:").pack(side=tk.LEFT)
        self.combo_optimizer = ttk.Combobox(
            opt_frame,
            state="readonly",
            values=[
                OptimizerFactory.MODE_PSO,
                OptimizerFactory.MODE_LINEAR,
                OptimizerFactory.MODE_HYBRID
            ],
            width=14
        )
        self.combo_optimizer.current(0)
        self.combo_optimizer.pack(side=tk.LEFT, padx=5)
        ToolTip(
            self.combo_optimizer,
            "PSO: 粒子群优化\nLinear Only: 直接线性最小二乘解\nLinear + PSO: 先线性解再 PSO 微调"
        )

        # 按钮行
        self.btn_start = ttk.Button(step_b, text="▶ 开始优化", command=self.start_opt)
        self.btn_start.pack(fill=tk.X, pady=3)
        self.btn_stop = ttk.Button(step_b, text="■ 停止", command=self.stop_opt, state=tk.DISABLED)
        self.btn_stop.pack(fill=tk.X)

        # 预览线性解
        btn_preview = ttk.Button(
            step_b,
            text="⚡ 预览线性解电压 (Lsq)",
            command=self.preview_linear_solution
        )
        btn_preview.pack(fill=tk.X, pady=3)
        ToolTip(
            btn_preview,
            "在不运行迭代优化的情况下，快速计算线性最小二乘解，"
            "用于评估目标可达性和初始电压分布。"
        )

        # --- Step C: 反馈校准 ---
        step_c = ttk.LabelFrame(frame_flow, text="Step C: 反馈校准", padding=2)
        step_c.pack(fill=tk.X, pady=5)

        self.btn_calibrate = ttk.Button(
            step_c,
            text="导入反馈数据并自动校准模型",
            command=self.perform_calibration,
            state=tk.DISABLED
        )
        self.btn_calibrate.pack(fill=tk.X, pady=5)
        ToolTip(
            self.btn_calibrate,
            "根据实测结果与理论预测的差异，\n自动修正每个电机的响应效率(Amps)。"
        )

        self.lbl_calib_status = ttk.Label(step_c, text="等待优化结果...", foreground="gray")
        self.lbl_calib_status.pack(anchor="w")

        # ================== 3. 结果分析 ==================
        frame_anl = ttk.LabelFrame(self.left_panel, text="3. 结果分析 (Analysis)", padding="5")
        frame_anl.pack(fill=tk.X, pady=5)

        self.btn_heatmap = ttk.Button(
            frame_anl,
            text="📊 查看电压迭代演变 (Heatmap)",
            command=self.show_voltage_heatmap,
            state=tk.DISABLED
        )
        self.btn_heatmap.pack(fill=tk.X, pady=2)

        self.txt_voltages = tk.Text(frame_anl, height=4, width=50)
        self.txt_voltages.pack(fill=tk.X, pady=2)

        ttk.Button(frame_anl, text="保存电压 CSV", command=self.save_results).pack(fill=tk.X)

        # ================== 4. 日志窗口 ==================
        frame_log = ttk.LabelFrame(self.left_panel, text="4. 日志 (Log)", padding="5")
        frame_log.pack(fill=tk.BOTH, pady=5, expand=True)

        self.txt_log = tk.Text(frame_log, height=12, width=50)
        self.txt_log.pack(fill=tk.BOTH, expand=True)
        self._append_log("程序启动。")

    def _add_input_row(self, parent, label_text, attr_name):
        ttk.Label(parent, text=label_text).pack(anchor="w")
        txt = tk.Text(parent, height=2, width=50)
        txt.pack(fill=tk.X)
        setattr(self, attr_name, txt)

    def _setup_plots(self):
        right_panel = ttk.Frame(self.root)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 创建 Figure
        self.fig = Figure(figsize=(10, 8), dpi=100)

        # 布局
        self.ax_curve = self.fig.add_subplot(221)
        self.ax_surf = self.fig.add_subplot(222)
        self.ax_volt = self.fig.add_subplot(223)
        self.ax_cost = self.fig.add_subplot(224)

        self.fig.tight_layout(pad=3.5)
        self.canvas = FigureCanvasTkAgg(self.fig, right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, right_panel)
        toolbar.update()

    # ================= 数据解析与默认值 =================
    def parse_array(self, widget):
        try:
            c = widget.get("1.0", tk.END).strip()
            if not c:
                return None
            return np.array(
                [float(x) for x in c.replace('\n', ',').split(',') if x.strip()]
            )
        except Exception:
            return None

    def fill_defaults(self):
        d_pos = "6.00, 15.65, 24.68, 32.61, 40.42, 48.50, 55.94, 64.78, 72.94, 80.72, 89.87, 98.16, 105.20, 112.54, 120.51, 129.29, 138.09, 151.26"
        d_wid = "18.40, 18.14, 14.88, 14.80, 15.63, 16.87, 18.70, 17.95, 17.43, 15.77, 18.27, 16.51, 17.13, 14.60, 16.55, 18.78, 17.45, 21.51"
        d_amp = "5.58, 5.99, 7.26, 7.66, 6.98, 6.75, 6.82, 7.31, 7.59, 7.40, 6.48, 7.08, 7.10, 7.98, 7.10, 5.57, 5.22, 8.29"

        self.txt_pos.delete("1.0", tk.END)
        self.txt_pos.insert(tk.END, d_pos)
        self.txt_width.delete("1.0", tk.END)
        self.txt_width.insert(tk.END, d_wid)
        self.txt_amp.delete("1.0", tk.END)
        self.txt_amp.insert(tk.END, d_amp)

    def _create_model(self):
        p = self.parse_array(self.txt_pos)
        w = self.parse_array(self.txt_width)
        a = self.parse_array(self.txt_amp)
        if p is None or w is None or a is None:
            messagebox.showerror("Err", "参数错误")
            return None
        try:
            model = StripDMModel(p, w, a * 1e-10)
            return model
        except Exception as e:
            messagebox.showerror("Err", str(e))
            return None

    # ================= 业务逻辑：基准与目标 =================
    def load_baseline_from_file(self):
        path = filedialog.askopenfilename(filetypes=[("Data", "*.csv *.txt")])
        if not path:
            return
        try:
            df = pd.read_csv(path, header=None, sep=None, engine='python')
            data = df.values
            if data.shape[1] < 2:
                raise ValueError("需要两列数据 (x, curvature)")

            data = data[data[:, 0].argsort()]
            self.baseline_x, self.baseline_y = data[:, 0], data[:, 1]

            self.lbl_baseline_status.config(
                text=f"状态: 已加载 ({len(data)} pts)",
                foreground="green"
            )
            self._append_log(f"加载基准面形: {path} ({len(data)} 点)")
            self.plot_preview()
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def clear_baseline(self):
        self.baseline_x, self.baseline_y = None, None
        self.lbl_baseline_status.config(text="状态: 未加载 (默认0)", foreground="red")
        self._append_log("清除基准面形。")
        self.plot_preview()

    def gen_target(self, mode):
        pos = self.parse_array(self.txt_pos)
        if pos is None:
            messagebox.showwarning("Warning", "请先设置有效的致动器位置参数。")
            return
        try:
            x, y = self.target_manager.generate(pos, mode=mode)
            self.target_x, self.target_y = x, y
            self._append_log(f"生成目标曲线: {mode}")
            self.plot_preview()
        except Exception as e:
            messagebox.showerror("Target Error", str(e))

    def load_target_from_file(self):
        path = filedialog.askopenfilename(filetypes=[("Data", "*.csv *.txt")])
        if not path:
            return
        try:
            x, y = self.target_manager.load_from_file(path)
            self.target_x, self.target_y = x, y
            self._append_log(f"从文件导入目标曲线: {path} ({len(x)} 点)")
            self.plot_preview()
        except Exception as e:
            messagebox.showerror("Target Load Error", str(e))

    def plot_preview(self):
        self.ax_curve.clear()
        self.ax_curve.set_title("Preview")
        if self.baseline_x is not None:
            self.ax_curve.plot(self.baseline_x, self.baseline_y, 'k--', label="Baseline")
        if hasattr(self, 'target_x') and self.target_x is not None:
            self.ax_curve.plot(self.target_x, self.target_y, 'r-', label="Target")
        self.ax_curve.legend()
        self.ax_curve.grid(True, alpha=0.3)
        self.canvas.draw()

    # ================= 业务逻辑：优化 =================
    def start_opt(self):
        self.dm_model = self._create_model()
        if not self.dm_model:
            return
        if not hasattr(self, 'target_x') or self.target_x is None:
            messagebox.showwarning("Warning", "请先设定目标 (Step B)")
            return

        # 准备基准 (插值到 target_x)
        base_interp = None
        if self.baseline_x is not None and self.baseline_y is not None:
            base_interp = np.interp(self.target_x, self.baseline_x, self.baseline_y)

        try:
            lim = float(self.entry_limit.get())
            iters = int(self.entry_iter.get())
            dvmax = float(self.entry_dvmax.get())
        except Exception:
            lim = 50.0
            iters = 200
            dvmax = 10.0

        self.target_iterations = iters
        self.safety_checker = VoltageSafetyChecker(limit_value=lim, max_delta_per_step=dvmax)

        # 优化模式
        self.current_optimizer_mode = self.combo_optimizer.get() if self.combo_optimizer else OptimizerFactory.MODE_PSO
        self.optimizer, self.linear_solver, run_mode = self.optimizer_factory.create(
            self.current_optimizer_mode,
            self.dm_model,
            self.target_x,
            self.target_y,
            baseline_y=base_interp,
            voltage_limit=lim,
            n_iter=iters,
            n_particles=40
        )

        self.is_running = True
        self.history = []

        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_calibrate.config(state=tk.DISABLED)
        self.btn_heatmap.config(state=tk.DISABLED)

        self._append_log(f"开始优化，模式: {run_mode}, Iter={iters}, Limit={lim:.1f} V, ΔVmax={dvmax:.1f} V")

        # 不同模式的执行逻辑
        if run_mode == OptimizerFactory.MODE_LINEAR:
            # 仅线性解，直接在主线程算完并更新界面
            self.run_linear_only(base_interp)
        elif run_mode == OptimizerFactory.MODE_HYBRID:
            # 先线性解，再以其为初值运行 PSO
            threading.Thread(target=self.run_hybrid_loop, args=(base_interp,), daemon=True).start()
        else:
            # 纯 PSO
            threading.Thread(target=self.run_loop, daemon=True).start()

    def stop_opt(self):
        self.is_running = False
        self._append_log("用户请求停止优化。")

    def run_linear_only(self, baseline_interp):
        try:
            if self.linear_solver is None:
                self.linear_solver = LinearLeastSquaresSolver(
                    self.dm_model, self.target_x, self.target_y,
                    baseline_y=baseline_interp,
                    voltage_limits=float(self.entry_limit.get())
                )
            v, merit = self.linear_solver.solve()
        except Exception as e:
            messagebox.showerror("Linear Solver Error", str(e))
            self._append_log(f"线性解失败: {e}")
            self.finish_opt()
            return

        self.history.append(merit)

        ok_lim, msg_lim = self.safety_checker.check_limits(v) if self.safety_checker else (True, "")
        ok_dv, msg_dv = self.safety_checker.check_delta(v, self.last_optimized_voltages) if self.safety_checker else (True, "")

        if not ok_lim or not ok_dv:
            messagebox.showwarning("安全检查警告", f"{msg_lim}\n{msg_dv}")
            self._append_log(f"安全检查未通过: {msg_lim}; {msg_dv}")

        # 在 GUI 主线程更新
        self.root.after(0, lambda: self.update_plots(v, merit))
        self.last_optimized_voltages = v.copy()
        self.last_target_x = self.target_x.copy()

        self._append_log(f"线性解完成: Merit={merit:.3e}")
        self.finish_opt()

    def run_hybrid_loop(self, baseline_interp):
        """
        先线性求解，再以线性解为初值跑 PSO
        """
        try:
            if self.linear_solver is None:
                self.linear_solver = LinearLeastSquaresSolver(
                    self.dm_model, self.target_x, self.target_y,
                    baseline_y=baseline_interp,
                    voltage_limits=float(self.entry_limit.get())
                )
            v0, merit0 = self.linear_solver.solve()
            self.optimizer.positions[0] = v0.copy()
            self.optimizer.gbest_pos = v0.copy()
            self.optimizer.gbest_score = merit0
            self._append_log(f"Hybrid: 线性初值 Merit={merit0:.3e}")
        except Exception as e:
            self._append_log(f"Hybrid 线性阶段失败: {e}")
            # 失败则退回纯 PSO 初始化
            pass

        # 再跑标准 PSO loop
        self.run_loop()

    def run_loop(self):
        for i in range(self.target_iterations):
            if not self.is_running:
                break
            score, best_v = self.optimizer.step()
            self.history.append(score)

            if i % 5 == 0 or i == self.target_iterations - 1:
                # 安全检查
                ok_lim, msg_lim = self.safety_checker.check_limits(best_v) if self.safety_checker else (True, "")
                ok_dv, msg_dv = self.safety_checker.check_delta(best_v, self.last_optimized_voltages) if self.safety_checker else (True, "")

                if not ok_lim or not ok_dv:
                    self._append_log(f"迭代 {i}: 安全检查未通过: {msg_lim}; {msg_dv}")
                    # 这里仅记录警告，不中断优化，真正写硬件前可以改为强制停止
                # 使用 after 在主线程更新 GUI
                self.root.after(
                    0,
                    lambda v=best_v.copy(), s=score: self.update_plots(v, s)
                )
                time.sleep(0.005)
        self.root.after(0, self.finish_opt)

    def update_plots(self, voltages, score):
        t_x = self.optimizer.target_x if self.optimizer is not None else self.target_x
        active_curve, _ = self.dm_model.get_active_curvature(t_x, voltages)

        if self.optimizer is not None:
            baseline = self.optimizer.baseline_y
            target_y = self.optimizer.target_y
        else:
            # 线性解模式
            if self.baseline_x is not None and self.baseline_y is not None:
                baseline = np.interp(t_x, self.baseline_x, self.baseline_y)
            else:
                baseline = np.zeros_like(t_x)
            target_y = self.target_y

        total_curve = baseline + active_curve

        # 1. Curvature
        self.ax_curve.clear()
        self.ax_curve.set_title(f"Curvature (Merit: {score:.2e})")
        self.ax_curve.plot(t_x, target_y, 'r-', lw=2, label="Target")
        self.ax_curve.plot(t_x, baseline, 'k--', alpha=0.4, label="Baseline")
        self.ax_curve.plot(t_x, total_curve, 'g-', label="Total")
        self.ax_curve.legend(fontsize='x-small')
        self.ax_curve.grid(True, linestyle='--', alpha=0.5)

        # 2. Surface Height (集成 RMS/PV 显示)
        slope = cumulative_trapezoid(total_curve, t_x, initial=0)
        height_mm = cumulative_trapezoid(slope, t_x, initial=0)
        height_nm = height_mm * 1e6

        # Detrend (Tilt Removal)
        p = np.polyfit(t_x, height_nm, 1)
        height_detrend = height_nm - np.polyval(p, t_x)

        # 计算统计指标
        rms_val = np.std(height_detrend)
        pv_val = np.ptp(height_detrend)  # peak to peak

        self.ax_surf.clear()
        self.ax_surf.set_title("Surface Height (Detrended)")
        self.ax_surf.plot(t_x, height_detrend, 'b-')
        self.ax_surf.set_ylabel("Height (nm)")
        self.ax_surf.grid(True, alpha=0.3)

        # 右上角显示参数框
        stats_text = f"RMS: {rms_val:.2f} nm\nPV:  {pv_val:.2f} nm"
        self.ax_surf.text(
            0.96, 0.96, stats_text,
            transform=self.ax_surf.transAxes,
            ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        )

        # 3. Voltage
        self.ax_volt.clear()
        self.ax_volt.set_title("Voltages (V)")
        self.ax_volt.bar(range(len(voltages)), voltages, color='skyblue', width=0.6)
        try:
            ylim = float(self.entry_limit.get())
        except Exception:
            ylim = np.max(np.abs(voltages)) * 1.1 + 1.0
        self.ax_volt.set_ylim(-ylim, ylim)

        # 4. Convergence
        self.ax_cost.clear()
        self.ax_cost.set_title("Merit Convergence")
        if len(self.history) > 0:
            self.ax_cost.semilogy(self.history)
        self.ax_cost.grid(True, alpha=0.3)

        self.canvas.draw()

    def finish_opt(self):
        self.is_running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

        if self.optimizer and self.optimizer.gbest_history:
            self.btn_heatmap.config(state=tk.NORMAL)  # 启用热力图按钮

        if self.optimizer:
            self.last_optimized_voltages = self.optimizer.gbest_pos.copy()
            self.last_target_x = self.optimizer.target_x.copy()
            v_final = self.optimizer.gbest_pos.copy()
        else:
            v_final = self.last_optimized_voltages

        if v_final is not None:
            v_str = ", ".join([f"{v:.3f}" for v in v_final])
            self.txt_voltages.delete("1.0", tk.END)
            self.txt_voltages.insert(tk.END, v_str)

            self.btn_calibrate.config(state=tk.NORMAL)
            self.lbl_calib_status.config(text="优化完成。如需校准请先测量反馈数据。", foreground="blue")

            # 自动保存结果
            meta = {
                "mode": self.current_optimizer_mode,
                "limit": self.entry_limit.get(),
                "iter": self.entry_iter.get(),
                "dvmax": self.entry_dvmax.get()
            }
            self.auto_saver.auto_save_results(v_final, meta=meta)
            self._append_log("优化完成，结果已自动保存 (若启用 AutoSave)。")

        messagebox.showinfo("完成", "优化结束。")

    # ================= 业务逻辑：线性解预览 =================
    def preview_linear_solution(self):
        """
        在当前目标和模型条件下，计算线性最小二乘解并更新图像，
        但不进入迭代优化过程。
        """
        self.dm_model = self._create_model()
        if not self.dm_model:
            return
        if not hasattr(self, 'target_x') or self.target_x is None:
            messagebox.showwarning("Warning", "请先设定目标 (Step B)")
            return

        base_interp = None
        if self.baseline_x is not None and self.baseline_y is not None:
            base_interp = np.interp(self.target_x, self.baseline_x, self.baseline_y)

        try:
            lim = float(self.entry_limit.get())
        except Exception:
            lim = 50.0

        try:
            solver = LinearLeastSquaresSolver(
                self.dm_model, self.target_x, self.target_y,
                baseline_y=base_interp,
                voltage_limits=lim
            )
            v, merit = solver.solve()
        except Exception as e:
            messagebox.showerror("Linear Preview Error", str(e))
            self._append_log(f"线性预览失败: {e}")
            return

        self._append_log(f"线性预览完成: Merit={merit:.3e}")
        # 临时更新图像（不改变历史和状态）
        self.update_plots(v, merit)

    # ================= 业务逻辑：热力图 =================
    def show_voltage_heatmap(self):
        """弹出窗口显示电压迭代热力图"""
        if not self.optimizer or not self.optimizer.gbest_history:
            messagebox.showinfo("Info", "无历史数据")
            return

        history = np.array(self.optimizer.gbest_history).T

        win = tk.Toplevel(self.root)
        win.title("Voltage Evolution Heatmap")
        win.geometry("800x600")

        fig_heat = Figure(figsize=(8, 6), dpi=100)
        ax = fig_heat.add_subplot(111)

        im = ax.imshow(
            history,
            aspect='auto',
            cmap='coolwarm',
            interpolation='nearest',
            vmin=-self.optimizer.limit_scalar,
            vmax=self.optimizer.limit_scalar
        )

        ax.set_xlabel("Iteration Step")
        ax.set_ylabel("Actuator Index")
        ax.set_title("Voltage Evolution during Optimization")

        cbar = fig_heat.colorbar(im, ax=ax)
        cbar.set_label("Voltage (V)")

        canvas_h = FigureCanvasTkAgg(fig_heat, win)
        canvas_h.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas_h, win)
        toolbar.update()

    # ================= 业务逻辑：校准 =================
    def perform_calibration(self):
        if self.last_optimized_voltages is None or self.baseline_x is None:
            messagebox.showerror("Err", "无历史数据或基准数据")
            return

        path = filedialog.askopenfilename(title="导入【反馈实测】曲率数据")
        if not path:
            return

        try:
            df = pd.read_csv(path, header=None, sep=None, engine='python')
            data = df.values
            data = data[data[:, 0].argsort()]
            new_meas_x, new_meas_y = data[:, 0], data[:, 1]

            # 对齐坐标系
            common_x = self.last_target_x
            base_interp = np.interp(common_x, self.baseline_x, self.baseline_y)
            new_interp = np.interp(common_x, new_meas_x, new_meas_y)

            real_delta = new_interp - base_interp

            # 执行校准
            logs = self.dm_model.calibrate_gains(
                self.last_optimized_voltages, real_delta, common_x
            )

            # 更新界面
            new_amps_display = self.dm_model.amps * 1e10
            self.txt_amp.delete("1.0", tk.END)
            self.txt_amp.insert(
                tk.END,
                ", ".join([f"{a:.5f}" for a in new_amps_display])
            )

            log_str = "\n".join(logs) if logs else "无显著调整。"
            win = tk.Toplevel(self.root)
            win.title("Calibration Report")
            t = tk.Text(win, height=20, width=80)
            t.pack()
            t.insert(tk.END, log_str)

            self._append_log(f"完成一次校准，条目数: {len(logs)}")

            # 更新基准
            self.baseline_x, self.baseline_y = new_meas_x, new_meas_y
            self.lbl_baseline_status.config(text="状态: 已更新为反馈数据", foreground="green")
            self.last_optimized_voltages = None
            self.btn_calibrate.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self._append_log(f"校准失败: {e}")

    def save_results(self):
        if self.last_optimized_voltages is None:
            messagebox.showinfo("Info", "暂无可保存的电压结果。")
            return
        f = filedialog.asksaveasfilename(defaultextension=".csv")
        if f:
            pd.DataFrame({"Voltage": self.last_optimized_voltages}).to_csv(f, index=False)
            self._append_log(f"手动保存电压结果到: {f}")

    # ================= 日志辅助 =================
    def _append_log(self, msg):
        line = self.log_manager.add(msg)
        if self.txt_log is not None:
            self.txt_log.insert(tk.END, line + "\n")
            self.txt_log.see(tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    # 针对高分屏优化（可选，Windows下有效）
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = StripDMApp(root)
    root.mainloop()
