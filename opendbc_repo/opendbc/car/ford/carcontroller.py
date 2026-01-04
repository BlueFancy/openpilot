import math
import cereal.messaging as messaging
import numpy as np
from numpy import clip, interp
from collections import deque
from common.filter_simple import FirstOrderFilter
from opendbc.can.packer import CANPacker
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, Bus, DT_CTRL, apply_std_steer_angle_limits, structs
from opendbc.car.vehicle_model import VehicleModel
from opendbc.car.ford import fordcan
from opendbc.car.ford.values import CarControllerParams, FordFlags, CAR
from opendbc.car.interfaces import CarControllerBase, V_CRUISE_MAX
from common.params import Params
from selfdrive.modeld.constants import ModelConstants  # 用于计算
from common.pid import PIDController # 横向控制的PID控制器
from bluepilot.params.bp_params import load_custom_params, update_custom_params  # 导入自定义参数函数
from opendbc.car.ford.helpers import compute_dm_msg_values
from bluepilot.logger.bp_logger import debug, info, warning, error, critical


LongCtrlState = structs.CarControl.Actuators.LongControlState
VisualAlert = structs.CarControl.HUDControl.VisualAlert

def index_function(idx, max_val=192, max_idx=32):
  return (max_val) * ((idx/max_idx)**2)

# ISO 11270 标准
ISO_LATERAL_ACCEL = 3.0  # 横向加速度，单位：m/s^2  # TODO: 从测试横向限制文件导入？

# 限制为平均倾斜道路，因为安全系统没有考虑侧倾
EARTH_G = 9.81  # 地球重力加速度
AVERAGE_ROAD_ROLL = 0.06  # 平均道路侧倾，约3.4度，6%超高
MAX_LATERAL_ACCEL = ISO_LATERAL_ACCEL - (EARTH_G * AVERAGE_ROAD_ROLL)  # 最大横向加速度，约2.4 m/s^2

# 模型预测变量
CONTROL_N = 17  # 控制点数
IDX_N = 33  # 索引数量
T_IDXS = [index_function(idx, max_val=10.0) for idx in range(IDX_N)]  # 时间索引数组


def apply_ford_curvature_limits(apply_curvature, apply_curvature_last, current_curvature, v_ego_raw, steering_angle, lat_active, CP):
  # 低速时不进行混合，因为缺乏扭矩缠绕且当前曲率不准确
  # 注意：
  # 福特Q3（非CAN-FD）可能会感觉像是在"等待"才开始转向，因为我们将请求的曲率
  # 限制在接近测量曲率（从yawRate推导）的范围内。在直路上，测量曲率在车辆实际
  # 开始转向之前一直保持在接近0，所以紧密的限制会强制命令"缓慢爬行"进入转弯。
  #
  # 为了保持直线稳定性，当请求较小时我们仍然紧密限制，但随着请求曲率的增长，
  # 我们*动态扩大*允许的误差窗口。这在不使直线驾驶振荡的情况下提高了转向响应速度。
  if v_ego_raw > 9:
    req_curv_mag = abs(apply_curvature)
    extra_err = float(np.interp(req_curv_mag,
                                # 曲率 [1/m]
                                [0.0, 0.004, 0.010, 0.020],
                                # 额外允许的误差 [1/m]
                                [0.0, 0.0015, 0.0035, 0.0060]))
    curvature_err = CarControllerParams.CURVATURE_ERROR + extra_err
    apply_curvature = np.clip(apply_curvature,
                              current_curvature - curvature_err,
                              current_curvature + curvature_err)

  # 在驾驶员扭矩限制后应用曲率变化速率限制
  apply_curvature = apply_std_steer_angle_limits(apply_curvature, apply_curvature_last, v_ego_raw, steering_angle, lat_active, CarControllerParams.ANGLE_LIMITS)

  # 福特Q4/CAN FD相比Q3/CAN有更多可用扭矩，因此我们基于横向加速度进行限制
  # 安全系统不知道道路侧倾，因此我们始终减去一个保守值
  if CP.flags & FordFlags.CANFD:
    # 将曲率限制为保守的最大横向加速度
    curvature_accel_limit = MAX_LATERAL_ACCEL / (max(v_ego_raw, 1) ** 2)
    apply_curvature = float(np.clip(apply_curvature, -curvature_accel_limit, curvature_accel_limit))

  return apply_curvature


def apply_creep_compensation(accel: float, v_ego: float) -> float:
  # 应用蠕变补偿：低速时补偿发动机蠕变
  creep_accel = np.interp(v_ego, [1., 3.], [0.6, 0.])  # 根据车速插值蠕变加速度
  creep_accel = np.interp(accel, [0., 0.2], [creep_accel, 0.])  # 根据加速度插值
  accel -= creep_accel  # 从加速度中减去蠕变补偿
  return float(accel)


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP, CP_SP):
    super().__init__(dbc_names, CP, CP_SP)

    self.params = Params()

    self.packer = CANPacker(dbc_names[Bus.pt])
    self.CAN = fordcan.CanBus(CP)


    # 从params.json加载初始自定义参数
    load_custom_params(self, "carcontroller")

    # 初始化控制变量
    self.apply_curvature_last = 0
    self.accel = 0.0
    self.gas = 0.0
    self.brake_request = False
    self.main_on_last = False
    self.lkas_enabled_last = False
    self.steer_alert_last = False
    self.fcw_alert_last = False  # 碰撞警报的上一次状态
    self.send_ui_last = False  # UI元素的上一次状态
    self.send_bars_ts_last = 0  # UI元素的上一次状态
    self.send_bars_last = False  # ACC间距元素的上一次状态
    self.lead_distance_bars_last = None
    self.distance_bar_frame = 0
    self.accel_pitch_compensated = 0.0
    self.steering_wheel_delta_adjusted = 0.0

   ################################## 横向控制参数 ##############################################

    # 需要初始化的变量（这些变量在每次扫描时作为控制代码的一部分进行更新）
    self.precision_type = 1  # 精度类型：1=精确模式，0=舒适模式
    self.human_turn = False  # 是否检测到驾驶员在转向时进行人工干预
    self.enable_lane_positioning = False # 从UI更新：启用高级车道定位
    self.enable_high_curvature_mode = False # 从UI更新：启用高曲率模式
    self.custom_profile = 0 # 从UI更新：自定义调优配置文件
    self.pc_blend_ratio = 0.5  # 预测曲率混合比例
    self.steer_warning = False # 转向限制超出的警告标志
    self.steer_warning_count = 0 # 警告存在的周期计数
    self.steering_limited = 0 # 转向被限制的周期计数

    # 曲率相关变量
    self.curvature_lookup_time = 0.15 # 曲率查找时间 - 减小以加快响应，更接近原车IPMA的即时性（从0.42改为0.15，减少延迟约64%）
    self.lane_change_factor_bp = [4.4, 40.23] # 调整车道变换因子的速度断点（m/s）
    self.lane_change_factor_low = 0.95 # 在4.4 m/s时的车道变换因子
    self.lane_change_factor_high = 0.85 # 从UI更新：在40.23 m/s时的车道变换因子
    self.pc_blend_ratio_low_C_CAN = 0.25 # CAN平台低曲率时的预测曲率混合比例（%） - 减小以更依赖当前期望曲率，减少延迟
    self.pc_blend_ratio_high_C_CAN = 0.30 # CAN平台高曲率时的预测曲率混合比例（%） - 减小以更依赖当前期望曲率
    self.pc_blend_ratio_low_C_CANFD = 0.25 # CAN-FD平台低曲率时的预测曲率混合比例（%） - 减小以更依赖当前期望曲率
    self.pc_blend_ratio_high_C_CANFD = 0.30 # CAN-FD平台高曲率时的预测曲率混合比例（%） - 减小以更依赖当前期望曲率
    self.pc_blend_ratio_low_C_UI = 0.25 # 从UI更新：低曲率时的预测曲率混合比例（%） - 减小以更依赖当前期望曲率
    self.pc_blend_ratio_high_C_UI = 0.30 # 从UI更新：高曲率时的预测曲率混合比例（%） - 减小以更依赖当前期望曲率
    self.pc_blend_ratio_bp = [0.0, 0.001] # 曲率断点，单位：1/m
    self.large_curve_factor_low = 1.0 # 小弯道时减少曲率的因子
    self.large_curve_factor_high = 0.90 # 大弯道时减少曲率的因子 - 增大以减少大弯道时的曲率削减，加快转向
    self.large_curve_factor_bp = [0.001, 0.02] # 曲率断点，单位：1/m
    self.large_curve_factor_v = [self.large_curve_factor_low, self.large_curve_factor_high] # 确定减少曲率的因子值

    # 曲率变化速率相关变量
    self.curvature_rate_delta_t = 0.2  # [秒] 用于曲率变化速率计算的分母 - 减小以加快响应（从0.3改为0.2）
    self.curvature_rate_deque = deque(maxlen=int(round(self.curvature_rate_delta_t / 0.05)))  # 在20Hz下0.2秒的队列
    self.curvature_rate_speed_bp = [0.0, 14.5, 15.5]  # 速度断点，单位：m/s
    self.curvature_rate_speed_v = [1.0, 1.0, 0.0]  # 对应的k_p值
    self.curvature_rate_PC_bp = [0.0, 0.003, 0.008, 0.01] # 曲率断点，单位：1/m - 降低阈值以在小曲率时也启用
    self.curvature_rate_PC_v = [0.3, 0.5, 0.8, 1.0] # 对应的k_p值 - 修改为在小曲率时也有值，而不是0

    # 路径偏移相关变量
    self.custom_path_offset = 0.0 # 从UI更新：应用自定义偏移以帮助车道内定位
    self.path_offset_lookup_time = 0.1 # 查找时间，单位：秒 - 减小以加快响应，更接近原车IPMA的即时性（从0.2改为0.1，减少延迟50%）
    self.lane_width_tolerance_factor = 0.75  # 车道宽度容差因子
    self.min_laneline_confidence_bp = [0.6, 0.8]  # 最小车道线置信度断点
    self.enable_lanefull_mode = True  # 启用完整车道模式

    # 路径角度共享变量
    self.path_angle_filter_samples = 3 # 移动平均滤波器使用的样本数量
    self.path_angle_deque = deque(maxlen=self.path_angle_filter_samples) # 用于保存样本的双端队列
    self.path_angle_wheel_angle_conversion = (np.pi/180) # 度到弧度的转换系数

    # 路径角度低曲率相关变量
    self.LC_PID_GAIN_CAN = 5.0  # CAN平台低曲率PID增益
    self.LC_PID_GAIN_CANFD_SMALL_VEHICLE = 3.0  # CAN-FD平台小型车辆低曲率PID增益
    self.LC_PID_GAIN_CANFD_LARGE_VEHICLE = 3.0  # CAN-FD平台大型车辆低曲率PID增益
    self.LC_PID_GAIN_UI = 0.0 # UI调优的增益
    self.LC_PID_GAIN = 0.0  # 当前使用的低曲率PID增益
    self.LC_PID_k_p = 0.25  # 低曲率PID比例系数
    self.LC_PID_k_i = 0.05  # 低曲率PID积分系数
    self.LC_PID_controller = PIDController(k_p=self.LC_PID_k_p, k_i=self.LC_PID_k_i, rate=20)  # 低曲率PID控制器
    self.LC_PID_speed_bp = [0.0, 9.0, 15.0]  # 速度断点，单位：m/s
    self.LC_PID_speed_v = [0.0, 0.0, 1.0]  # 对应的k_p值
    self.LC_path_angle_ROC_bp = [5, 15, 25]  # 路径角度变化速率的速度断点，单位：m/s
    self.LC_path_angle_ROC_v = [0.003, 0.0015, 0.002]  # 匹配panda限制
    self.LC_path_angle_reset_counter = 0  # 路径角度重置计数器
    self.LC_path_angle_reset_duration = 1.5 # 重置持续时间，单位：秒

    # 路径角度高曲率相关变量
    self.HC_PID_gain_UI = 0.5 # UI调优的增益
    self.HC_PID_k_p = 1.0  # 高曲率PID比例系数
    self.HC_PID_k_i = 0.05  # 高曲率PID积分系数
    self.HC_PID_controller = PIDController(k_p=self.HC_PID_k_p, k_i=self.HC_PID_k_i, rate=20)  # 高曲率PID控制器
    self.wheel_angle_lookup_time = 0.02  # 车轮角度查找时间 - 减小以加快响应（从0.05改为0.02，减少延迟60%）
    self.HC_PID_curvature_bp = [0.0, 0.008, 0.01, 0.02]  # 曲率断点，单位：1/m
    self.HC_PID_curvature_v = [0.0, 0.0, 1.0, 1.0]  # 对应的k_p值
    self.HC_PID_speed_bp = [0.0, 20.00, 22.00, 25.00]  # 调整path_angle_speed_factor的速度断点
    self.HC_PID_speed_v = [1.0, 1.0, 0.0, 0.0]  # 对应的速度因子值
    self.pswa_blend_ratio = 1.0  # 预测转向角度混合比例

    # 所有四个信号的最大绝对值
    self.path_angle_max = 0.5  # 来自dbc文件：路径角度最大值
    self.path_offset_max = 2.0  # 路径偏移最大值：过多的路径偏移会导致问题
    self.curvature_max = 0.02  # 来自dbc文件：曲率最大值
    self.curvature_rate_max = 0.002  # 曲率变化速率最大值 - 增大以允许更快的转向（从0.001023改为0.002，约+95%）

    # 上一帧的值
    self.curvature_rate_last = 0.0  # 上一帧的曲率变化速率
    self.path_offset_last = 0.0  # 上一帧的路径偏移
    self.path_angle_last = 0.0  # 上一帧的路径角度
    self.curvature_rate = 0  # 初始化曲率变化速率


    # 日志记录变量
    debug(f'Car Fingerprint (CarController): {CP.carFingerprint}', True)

    # 车道变换过渡跟踪
    self.post_lane_change_timer = 0
    self.post_lane_change_active = False
    self.lane_change_last = False  # 跟踪上一次的车道变换状态
    self.pre_lane_change_values = {
        'path_angle': 0.0,
        'path_offset': 0.0,
        'desired_curvature_rate': 0.0
    }

    # 每帧允许的最大变化量
    self.max_path_angle_change = 0.00125  # 路径角度最大变化
    self.max_path_offset_change = 0.00125  # 路径偏移最大变化
    self.max_curvature_rate_change = 0.0001  # 曲率变化速率最大变化

    self.sm = messaging.SubMaster(['modelV2', 'liveParameters', 'selfdriveState'])
    self.VM = VehicleModel(self.CP)
    self.curvature_lookup_time = 0.15  # 与上面的curvature_lookup_time保持一致，减小延迟以更接近原车IPMA的即时性

    self.model = None
    self.lp = None
    self.ss = None
    self.send_driver_monitor_can_msg = False
    self.send_lane_depart_can_msg = False
    self.send_hands_free_cluster_msg = False
    self.tja_msg = 0
    self.tja_warn = 0
    self.hands = 0
    self.predictedSteeringAngleDeg_SP = 0.0

  def handle_post_lane_change_transition(self, path_angle, path_offset, desired_curvature_rate):
    """
    管理车道变换后控制变量的平滑过渡
    返回: (path_angle, path_offset, desired_curvature_rate) 元组
    """
    # 检测车道变换完成（从True过渡到False）
    if self.lane_change_last and not self.lane_change:
        self.post_lane_change_active = True
        self.post_lane_change_timer = 0
        # 将当前值存储为起始点
        self.pre_lane_change_values = {
            'path_angle': 0.0,  # 从零开始，因为我们正在退出车道变换
            'path_offset': 0.0,
            'desired_curvature_rate': 0.0
        }

    # 更新上一次的车道变换状态
    self.lane_change_last = self.lane_change

    # 如果我们在车道变换后状态
    if self.post_lane_change_active:
        self.post_lane_change_timer += 1

        # 使用速率限制应用平滑过渡
        new_path_angle = clip(
            path_angle,
            self.pre_lane_change_values['path_angle'] - self.max_path_angle_change,
            self.pre_lane_change_values['path_angle'] + self.max_path_angle_change
        )

        new_path_offset = clip(
            path_offset,
            self.pre_lane_change_values['path_offset'] - self.max_path_offset_change,
            self.pre_lane_change_values['path_offset'] + self.max_path_offset_change
        )

        new_curvature_rate = clip(
            desired_curvature_rate,
            self.pre_lane_change_values['desired_curvature_rate'] - self.max_curvature_rate_change,
            self.pre_lane_change_values['desired_curvature_rate'] + self.max_curvature_rate_change
        )

        # 更新存储的值
        self.pre_lane_change_values = {
            'path_angle': new_path_angle,
            'path_offset': new_path_offset,
            'desired_curvature_rate': new_curvature_rate
        }

        # 在160帧后退出过渡状态
        if self.post_lane_change_timer >= 160:
            self.post_lane_change_active = False

        return (new_path_angle, new_path_offset, new_curvature_rate)

    return (path_angle, path_offset, desired_curvature_rate)

  def update(self, CC, CC_SP, CS, now_nanos):
    can_sends = []
    self.sm.update(0)

    if self.sm.updated['modelV2']:
      self.model = self.sm["modelV2"]

    if self.sm.updated['liveParameters']:
      self.lp = self.sm["liveParameters"]

    if self.sm.updated['selfdriveState']:
      self.ss = self.sm['selfdriveState']

    if self.lp is not None:
      x = max(self.lp.stiffnessFactor, 0.1)
      sr = max(self.lp.steerRatio, 0.1)
      self.VM.update_params(x, sr)

    # 触发设置参数的更新
    # update_settings_params(self)
    update_custom_params(self, "carcontroller")

    actuators = CC.actuators
    hud_control = CC.hudControl
    main_on = CS.out.cruiseState.available
    # if self.fordVariables is None:
      # act = actuators.as_builder()
      # self.fordVariables = act.fordVariables

    # 计算转向警报和前向碰撞警告
    steer_alert = False
    fcw_alert = hud_control.visualAlert == VisualAlert.fcw

    # 计算驾驶员监控消息值
    if self.send_driver_monitor_can_msg:
      # print(f'HudControl: {hud_control}')
      # print(f'tja_msg: {self.tja_msg} | tja_warn: {self.tja_warn}')
      if (self.frame % CarControllerParams.ACC_UI_STEP) == 0:
        self.tja_msg, self.tja_warn, self.hands = compute_dm_msg_values(self.ss, hud_control, self.send_hands_free_cluster_msg, main_on, CS.out.cruiseState.standstill)
    else:
      steer_alert = hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw)
      if steer_alert:
        self.hands = 1
      else:
        self.hands = 0

    ### ACC按钮 ###
    if CC.cruiseControl.cancel:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, cancel=True))
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.main, CS.buttons_stock_values, cancel=True))
    elif CC.cruiseControl.resume and (self.frame % CarControllerParams.BUTTONS_STEP) == 0:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, resume=True))
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.main, CS.buttons_stock_values, resume=True))
    # 如果原厂车道居中未关闭，发送按钮按下以切换关闭
    # 原厂系统检查转向是否被按下，最终会解除巡航控制
    elif CS.acc_tja_status_stock_values["Tja_D_Stat"] != 0 and (self.frame % CarControllerParams.ACC_UI_STEP) == 0:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, tja_toggle=True))

    ### 横向控制 ###

    apply_curvature = 0.0 # 初始化应用曲率
    desired_curvature_rate = 0.0 # 初始化期望曲率变化速率
    path_offset = 0.0 # 初始化路径偏移
    path_angle = 0.0 # 初始化路径角度
    reset_steering = 0 # 初始化重置转向标志
    ramp_type = 2 # 初始化斜坡类型（2=快速模式）

    # 以20Hz频率发送转向消息
    if (self.frame % CarControllerParams.STEER_STEP) == 0:
      if CC.latActive:
        self.precision_type = 1  # 设置为精确模式
        steeringPressed = CS.out.steeringPressed  # 驾驶员是否按下转向
        steeringAngleDeg_PV = CS.out.steeringAngleDeg  # 车辆当前转向角度
        steeringAngleDeg_SP = actuators.steeringAngleDeg  # 期望的转向角度

        # 确定调优配置文件
        if self.custom_profile == 1: # 自定义调优配置文件
          self.pc_blend_ratio_low_C =  self.pc_blend_ratio_low_C_UI
          self.pc_blend_ratio_high_C =  self.pc_blend_ratio_high_C_UI
          self.LC_PID_GAIN = self.LC_PID_GAIN_UI

        elif self.CP.flags & FordFlags.CANFD:
          self.pc_blend_ratio_low_C = self.pc_blend_ratio_low_C_CANFD
          self.pc_blend_ratio_high_C = self.pc_blend_ratio_high_C_CANFD
          if (self.CP.carFingerprint == CAR.FORD_ESCAPE_MK5 or self.CP.carFingerprint == CAR.FORD_MUSTANG_MACH_E_MK1):
            self.LC_PID_gain = self.LC_PID_GAIN_CANFD_SMALL_VEHICLE
          else:
            self.LC_PID_gain = self.LC_PID_GAIN_CANFD_LARGE_VEHICLE
        else:
          self.pc_blend_ratio_low_C = self.pc_blend_ratio_low_C_CAN
          self.pc_blend_ratio_high_C = self.pc_blend_ratio_high_C_CAN
          self.LC_PID_gain = self.LC_PID_GAIN_CAN

        self.pc_blend_ratio_v = [self.pc_blend_ratio_low_C, self.pc_blend_ratio_high_C] # 预测曲率混合比例值数组

        # 计算当前曲率和模型期望曲率
        current_curvature = -CS.out.yawRate / max(CS.out.vEgoRaw, 0.1)  # 使用CAN总线数据计算当前曲率
        desired_curvature = actuators.curvature  # 从模型获取期望曲率

        # 从modelV2提取预测曲率
        if self.model is not None and len(self.model.orientation.x) >= 17:
          # 从模型预测的orientationRate计算曲率，并根据最大预测曲率幅度与期望曲率混合
          curvatures = np.array(self.model.orientationRate.z) / max(0.01, CS.out.vEgoRaw)
          predicted_steering_angle_curvature = interp(self.wheel_angle_lookup_time, ModelConstants.T_IDXS, curvatures)
          predicted_curvature = interp(self.curvature_lookup_time, ModelConstants.T_IDXS, curvatures)
          max_abs_predicted_curvature = max(np.abs(curvatures[:17]))  # 未来2.5秒内的最大曲率幅度
        else:
          predicted_curvature = 0.0

        # 计算预测转向角度
        self.predictedSteeringAngleDeg_SP = math.degrees(self.VM.get_steer_from_curvature(-predicted_steering_angle_curvature, CS.out.vEgoRaw, 0))
        self.predictedSteeringAngleDeg_SP += self.lp.angleOffsetDeg

        # 计算混合比例
        self.pc_blend_ratio = interp(abs(desired_curvature), self.pc_blend_ratio_bp, self.pc_blend_ratio_v)

        # 将请求曲率设置为期望曲率和预测曲率的混合，并应用曲率限制
        requested_curvature = (predicted_curvature * self.pc_blend_ratio) + (desired_curvature * (1 - self.pc_blend_ratio))

        # 确定是否正在进行车道变换
        if (self.model.meta.laneChangeState == 1 or self.model.meta.laneChangeState == 2 or self.model.meta.laneChangeState == 3):
            self.lane_change = True
        else:
            self.lane_change = False

        # 根据速度确定车道变换因子
        lane_change_factor = interp(CS.out.vEgoRaw, self.lane_change_factor_bp, [self.lane_change_factor_low, self.lane_change_factor_high])

        # 如果正在变换车道，修改曲率以平滑车道变换
        if self.lane_change and (self.model.meta.laneChangeDirection == 1): # 如果正在向左变换车道
          if requested_curvature < 0: # 且曲率正在引导我们向左
              requested_curvature = requested_curvature * lane_change_factor # 减小曲率以平滑车道变换
          else:
              requested_curvature = requested_curvature # 如果正在向右回正以纠正过度转向，则不减小曲率

          self.precision_type = 0 # 使用舒适模式

        if self.lane_change and (self.model.meta.laneChangeDirection == 2): # 如果正在向右变换车道
          if requested_curvature > 0: # 且曲率正在引导我们向右
              requested_curvature = requested_curvature * lane_change_factor # 减小曲率以平滑车道变换
          else:
              requested_curvature = requested_curvature # 如果正在向左回正以纠正过度转向，则不减小曲率

          self.precision_type = 0 # 使用舒适模式

        # 应用曲率限制
        apply_curvature = apply_ford_curvature_limits(requested_curvature,
                                                                self.apply_curvature_last,
                                                                current_curvature,
                                                                CS.out.vEgoRaw,
                                                                0,
                                                                CC.latActive,
                                                                self.CP)


        # 检测转向是否被限制（车道变换总是会触发，但能正常完成）
        if (requested_curvature != apply_curvature) and (not steeringPressed) and (not self.lane_change):
          self.steering_limited = self.steering_limited + 1
        else:
          self.steering_limited = 0

        # 如果转向被限制超过10次扫描且速度高于15mph，则开启转向警告
        if self.steering_limited > 10 and CS.out.vEgoRaw > 7:
            self.steer_warning = True

        # 锁定转向警告并计数周期，然后清除
        if self.steer_warning and not self.steering_limited:
            self.steer_warning_count = self.steer_warning_count + 1

        # 在10次无转向限制计数后清除转向警告
        if self.steer_warning_count > 10:
          self.steer_warning = False
          self.steer_warning_count = 0

        # 计算曲率变化速率
        self.curvature_rate_deque.append(predicted_curvature)
        if len(self.curvature_rate_deque) > 1:
          delta_t = (
            self.curvature_rate_delta_t if len(self.curvature_rate_deque) == self.curvature_rate_deque.maxlen else (len(self.curvature_rate_deque) - 1) * 0.05
          )
          desired_curvature_rate = (self.curvature_rate_deque[-1] - self.curvature_rate_deque[0]) / delta_t / max(0.01, CS.out.vEgoRaw)
        else:
          desired_curvature_rate = 0.0

        # 计算曲率变化速率的预测曲率因子
        curvature_rate_PC_factor = interp(abs(predicted_curvature), self.curvature_rate_PC_bp, self.curvature_rate_PC_v)
        desired_curvature_rate = desired_curvature_rate * curvature_rate_PC_factor

        # 计算曲率变化速率的速度因子
        curvature_rate_speed_factor = interp(CS.out.vEgoRaw, self.curvature_rate_speed_bp, self.curvature_rate_speed_v)
        desired_curvature_rate = desired_curvature_rate * curvature_rate_speed_factor

        # 确定大弯道因子
        large_curve_factor = interp(abs(requested_curvature), self.large_curve_factor_bp, self.large_curve_factor_v)

        # 将大弯道因子应用到期望曲率变化速率
        desired_curvature_rate = desired_curvature_rate * large_curve_factor

        # 车道变换时不使用大弯道因子
        if self.lane_change:
          large_curve_factor = 1.0

        # 如果正在进行车道变换，将期望曲率变化速率设置为0
        if self.lane_change:
          desired_curvature_rate = 0.0

        # 确定驾驶员是否正在转向并捕获该值
        if steeringPressed and abs(steeringAngleDeg_PV) > 45:
          self.human_turn = True
        else:
          self.human_turn = False

        # 从model.position.y获取路径偏移
        path_offset_position = interp(self.path_offset_lookup_time, ModelConstants.T_IDXS, self.model.position.y)

        # 现在从车道线获取路径偏移
        path_offset_lanelines = (self.model.laneLines[1].y[0] + self.model.laneLines[2].y[0]) / 2

        # 确定车道线宽度容差缩放因子
        laneline_width = self.model.laneLines[2].y[0] + (-self.model.laneLines[1].y[0]) # laneLines[1]是负值，因为它在车辆左侧
        laneline_width_tolerance = interp(laneline_width, [3.75,4.25], [0.81, 0.59]) # 3.7米是美国标准车道宽度

        # 确定车道线置信度
        laneline_confidence = min(self.model.laneLineProbs[1], self.model.laneLineProbs[2], laneline_width_tolerance)
        if not self.enable_lanefull_mode:
          laneline_confidence = 0.0

        # 确定车道线路径偏移缩放比例
        laneline_path_offset_scale = interp(laneline_confidence, self.min_laneline_confidence_bp, [0.0, 1.0])

        # 获取结合模型和车道线的总路径偏移
        path_offset = (path_offset_position * (1-laneline_path_offset_scale) + (path_offset_lanelines * laneline_path_offset_scale)) + self.custom_path_offset

        # 车道变换期间不使用路径偏移（如果不设置为零，它会一直对抗直到切换到新车道）
        if self.lane_change:
          path_offset = 0

        # 使用UI变量进行可调增益，将PID增益设置为固定值，UI变量除以100以使UI变量更接近2.1逻辑调优
        path_offset_error = (path_offset * (self.LC_PID_gain_UI/100))

        # 确定速度因子
        LC_PID_speed_factor = interp(CS.out.vEgoRaw, self.LC_PID_speed_bp, self.LC_PID_speed_v)

        # 将速度因子应用到path_offset_error
        path_offset_error_adj = path_offset_error * LC_PID_speed_factor

        # 如果不使用车道定位，将path_offset_error_adj置零
        if not self.enable_lane_positioning:
          path_offset_error_adj = 0.0

        # 使用path_angle帮助车辆在车道中居中，使用PID控制器计算path_angle
        path_angle_low_c = self.LC_PID_controller.update(path_offset_error_adj)

        # 如果不使用车道定位，将path_angle_low_c置零（应该在PID控制器中置零，但以防万一）
        if not self.enable_lane_positioning:
          path_angle_low_c = 0.0

        # 对path_angle_low_c进行速率限制以提高舒适性
        path_angle_roc = interp(abs(CS.out.vEgoRaw), self.LC_path_angle_ROC_bp, self.LC_path_angle_ROC_v)
        path_angle_low_c = clip(path_angle_low_c, self.path_angle_last - path_angle_roc, self.path_angle_last + path_angle_roc)

        # 如果驾驶员持续对方向盘施加压力，重置path_angle_low_c PID控制器
        if steeringPressed:
          self.LC_path_angle_reset_counter = self.LC_path_angle_reset_counter + 1
        else:
          self.LC_path_angle_reset_counter = 0
        if self.LC_path_angle_reset_counter > self.LC_path_angle_reset_duration * 20: #每秒20次扫描
          self.LC_PID_controller.reset()

        # path_angle是一个修正变量，因此减去当前车轮位置（与曲率相关）
        # 计算方向盘距离期望转向角度的偏差
        steering_wheel_delta = (steeringAngleDeg_PV - self.predictedSteeringAngleDeg_SP) * self.path_angle_wheel_angle_conversion

        # 计算高曲率情况下的路径角度：
        # 根据曲率插值path_angle_factor
        HC_PID_curvature_factor = interp(max_abs_predicted_curvature, self.HC_PID_curvature_bp, self.HC_PID_curvature_v)

        # 根据速度插值path_angle_speed_factor
        HC_PID_speed_factor = interp(CS.out.vEgoRaw, self.HC_PID_speed_bp, self.HC_PID_speed_v)

        # 选择使用哪个path_angle因子（曲率或速度）
        HC_PID_adjust_factor = min(HC_PID_curvature_factor, HC_PID_speed_factor)

        # 在执行PID之前将调整因子应用到steering_wheel_delta
        self.steering_wheel_delta_adjusted = steering_wheel_delta * HC_PID_adjust_factor * self.HC_PID_gain_UI

        # 如果不使用高曲率模式，将steering_wheel_delta_adjusted置零
        if not self.enable_high_curvature_mode:
          self.steering_wheel_delta_adjusted = 0.0

        # 使用PID计算path_angle，增益根据曲率插值（默认根据速度）
        path_angle_high_c = self.HC_PID_controller.update((self.steering_wheel_delta_adjusted))

        # 如果不使用高曲率模式，将path_angle_high_c置零（应该在PID控制器中置零，但以防万一）
        if not self.enable_high_curvature_mode:
          path_angle_high_c = 0.0

        # 将path_angle_low_c和path_angle_high_c相加
        path_angle = path_angle_low_c + path_angle_high_c

        # 车道变换期间将path_angle置零
        if self.lane_change:
          path_angle = 0.0

        # 如果path_angle信号被因子置零，重置PID控制器
        if HC_PID_adjust_factor < 0.1:
          self.HC_PID_controller.reset()

        # 对path_angle进行速率限制（已注释）
        # path_angle_roc = interp(abs(CS.out.vEgoRaw), [5, 25], [0.003, 0.002])
        # path_angle = clip(path_angle, self.path_angle_last - path_angle_roc, self.path_angle_last + path_angle_roc)

        # 应用车道变换后过渡逻辑
        path_angle, path_offset, desired_curvature_rate = self.handle_post_lane_change_transition(
            path_angle, path_offset, desired_curvature_rate
        )

        # 将所有值限制在最大值范围内
        apply_curvature = clip(apply_curvature, -self.curvature_max, self.curvature_max)
        desired_curvature_rate = clip(desired_curvature_rate, -self.curvature_rate_max, self.curvature_rate_max)
        path_offset = clip(path_offset, -self.path_offset_max, self.path_offset_max)
        path_angle = clip(path_angle, -self.path_angle_max, self.path_angle_max)


        # 如果path_offset和path_angle不一致，可能导致非常不舒适的驾驶，因为path_angle很强，在通过canbus发送之前将path_offset信号置零
        path_offset = 0.0

        # 确定驾驶员是否正在转向并捕获该值
        # 如果检测到人工转向，重置转向以防止缠绕
        if steeringPressed and abs(steeringAngleDeg_PV) > 45:
          self.human_turn = True
        else:
          self.human_turn = False

        # 确定何时重置转向
        if ((self.human_turn) and self.enable_human_turn_detection) or (CS.out.vEgoRaw < 0.1):
          reset_steering = 1
        else:
          reset_steering = 0

        # 通过将所有值设置为0并将ramp_type设置为立即来重置转向
        if reset_steering == 1:
          apply_curvature = 0
          path_offset = 0
          path_angle = 0
          desired_curvature_rate = 0
          ramp_type = 3  # 立即模式
          self.path_angle_deque.clear()
          self.HC_PID_controller.reset()
          self.LC_PID_controller.reset()
        else:
          ramp_type = 2  # 快速模式
      else:
        # 横向控制未激活时，将所有值置零
        apply_curvature = 0.0
        desired_curvature_rate = 0.0
        path_offset = 0.0
        path_angle = 0.0
        self.path_angle_deque.clear()
        self.HC_PID_controller.reset()
        self.LC_PID_controller.reset()
        ramp_type = 0  # 无模式

      self.apply_curvature_last = apply_curvature
      self.curvature_rate_last = desired_curvature_rate
      self.path_offset_last = path_offset
      self.path_angle_last = path_angle


      # 将lat_active设置为CC.latActive的值
      lat_active = CC.latActive

      if self.CP.flags & FordFlags.CANFD:
        # TODO: 扩展模式
        # 福特使用四个独立信号来控制车辆驾驶。仅曲率（限制为0.02m/s^2）
        # 可以驱动大部分横向运动的转向。然而，为了获得对转向驱动的进一步控制，
        # 其他三个信号是必要的。福特控制车辆的方式与大多数其他品牌不同。
        # 关于福特控制的详细说明可以在这里找到：
        # https://www.f150gen14.com/forum/threads/introducing-bluepilot-a-ford-specific-fork-for-comma3x-openpilot.24241/#post-457706
        mode = 1 if lat_active else 0
        counter = (self.frame // CarControllerParams.STEER_STEP) % 0x10
        can_sends.append(fordcan.create_lat_ctl2_msg(
          self.packer, self.CAN, mode, ramp_type, self.precision_type, -path_offset, -path_angle,
          -apply_curvature, -desired_curvature_rate, counter
        ))
      else:
        # 福特非CAN-FD横向控制
        can_sends.append(fordcan.create_lat_ctl_msg(
          self.packer, self.CAN, lat_active, ramp_type, self.precision_type,
          -path_offset, -path_angle, -apply_curvature, -desired_curvature_rate
        ))

    # 以33Hz频率发送LKA消息
    if (self.frame % CarControllerParams.LKA_STEP) == 0:
      lka_hud_control = None
      if self.send_lane_depart_can_msg:
        lka_hud_control = hud_control
      can_sends.append(fordcan.create_lka_msg(self.packer, self.CAN, CC.latActive, lka_hud_control))

    ### 纵向控制 ###
    # 以50Hz频率发送ACC消息
    if self.CP.openpilotLongitudinalControl and (self.frame % CarControllerParams.ACC_CONTROL_STEP) == 0:
      accel = actuators.accel
      gas = accel

      if CC.longActive:
        # 低速时补偿发动机蠕变
        # ABS要么没有考虑发动机蠕变，要么修正非常慢
        # TODO: 验证这适用于EV/混合动力车
        accel = apply_creep_compensation(accel, CS.out.vEgo)

        # 原厂系统已被观察到将刹车加速度限制为5 m/s^3，
        # 然而即使是3.5 m/s^3也会在阶跃响应中引起一些超调
        accel = max(accel, self.accel - (3.5 * CarControllerParams.ACC_CONTROL_STEP * DT_CTRL))

      accel = float(np.clip(accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))
      gas = float(np.clip(gas, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))

      # 油门和加速度都以m/s^2为单位，accel仅用于刹车
      if not CC.longActive or gas < CarControllerParams.MIN_GAS:
        gas = CarControllerParams.INACTIVE_GAS

      # PCM对油门/加速度应用俯仰补偿，但我们需要补偿刹车/预充位
      accel_due_to_pitch = 0.0
      if len(CC.orientationNED) == 3:
        accel_due_to_pitch = math.sin(CC.orientationNED[1]) * ACCELERATION_DUE_TO_GRAVITY

      accel_pitch_compensated = accel + accel_due_to_pitch
      if accel_pitch_compensated > 0.3 or not CC.longActive:
        self.brake_request = False
      elif accel_pitch_compensated < 0.0:
        self.brake_request = True

      stopping = CC.actuators.longControlState == LongCtrlState.stopping
      # TODO: 研究使用actuators数据包发送期望速度
      can_sends.append(fordcan.create_acc_msg(self.packer, self.CAN, CC.longActive, gas, accel, stopping, self.brake_request, v_ego_kph=V_CRUISE_MAX))

      self.accel = accel
      self.gas = gas
      self.accel_pitch_compensated = accel_pitch_compensated

    ### 用户界面 ###
    send_ui = (self.main_on_last != main_on) or (self.lkas_enabled_last != CC.latActive) or (self.steer_alert_last != steer_alert)
    # 以1Hz频率发送LKAS UI消息，或在UI状态改变时发送
    if (self.frame % CarControllerParams.LKAS_UI_STEP) == 0 or send_ui:
      can_sends.append(fordcan.create_lkas_ui_msg(self.packer, self.CAN, main_on, CC.latActive, self.hands, hud_control, CS.lkas_status_stock_values))

    # 以5Hz频率发送ACC UI消息，或在UI状态改变时发送
    send_bars = False
    if hud_control.leadDistanceBars != self.lead_distance_bars_last:
      send_ui = True
      send_bars = True

    # 保持发送距离条的逻辑，持续4秒
    if not self.send_bars_last and send_bars:
      # 保存从False翻转到True的最后一帧编号
      self.send_bars_ts_last = self.frame
      self.distance_bar_frame = self.frame

    # 保持发送距离条4秒（在100Hz下为400帧）
    if (self.send_bars_ts_last > 0 and (self.frame - self.send_bars_ts_last) <= 400):
      send_ui = True
      send_bars = True

    if (self.frame % CarControllerParams.ACC_UI_STEP) == 0 or send_ui:
      can_sends.append(
        fordcan.create_acc_ui_msg(
          self.packer,
          self.CAN,
          self.CP,
          main_on,
          CC.latActive,
          fcw_alert,
          CS.out.cruiseState.standstill,
          hud_control,
          CS.acc_tja_status_stock_values,
          self.send_hands_free_cluster_msg,
          send_ui,
          send_bars,
          self.tja_warn,
          self.tja_msg,
        )
      )

    self.main_on_last = main_on
    self.send_ui_last = send_ui
    self.send_bars_last = send_bars
    self.lkas_enabled_last = CC.latActive
    self.steer_alert_last = steer_alert
    self.fcw_alert_last = fcw_alert
    self.lead_distance_bars_last = hud_control.leadDistanceBars

    new_actuators = actuators.as_builder()
    new_actuators.torqueOutputCan = float(self.steer_warning)
    new_actuators.curvature = float(apply_curvature)
    new_actuators.accel = float(self.accel)
    new_actuators.gas = float(self.gas)
    new_actuators.steeringAngleDeg = float(self.predictedSteeringAngleDeg_SP)
    self.frame += 1
    return new_actuators, can_sends
