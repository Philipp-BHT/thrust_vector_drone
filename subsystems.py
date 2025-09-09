import numpy as np
import math

import math

class Motor:
    # Parameters assuming 6040 2-blade propellers
    MOTOR_PARAMETERS = {
        "T-Motor F80 PRO 2408 Brushless Motor": {
            "max_rpm": 29613,
            "voltage": 18.79,   # V
            "current": 49.40,   # A
            "power": 928.25,    # W
            "max_thrust": 2037, # g  (datasheet uses grams)
            "power_ratio": 2.19,# g/W
            "propeller_radius": 15.24 # cm (6 inches)
        }
    }

    def __init__(self, model: str = "T-Motor F80 PRO 2408 Brushless Motor", tau_motor=0.18):
        p = self.MOTOR_PARAMETERS[model]
        self.omega_max = p["max_rpm"] * 2*math.pi / 60.0   # rad/s
        self.T_max_N   = p["max_thrust"] * 9.81 / 1000.0   # g -> N
        self.tau       = tau_motor                         # first-order time constant [s]

        # Motor internal state
        self.omega     = 0.0       # achieved speed [rad/s]
        self.omega_cmd = 0.0       # commanded speed [rad/s]

    # --- Control interface (choose ONE to use) ---
    def set_throttle(self, u: float):
        """u in [0..1] → target speed."""
        u = max(0.0, min(1.0, u))
        self.omega_cmd = u * self.omega_max

    def set_target_speed(self, omega_target: float):
        """Direct speed command [rad/s]."""
        self.omega_cmd = max(0.0, min(self.omega_max, omega_target))

    # --- Plant update ---
    def update(self, dt: float):
        """First-order lag to commanded speed; clamp to physical max."""
        if self.tau <= 0.0:
            self.omega = self.omega_cmd
        else:
            self.omega += (self.omega_cmd - self.omega) * (dt / self.tau)
        self.omega = max(0.0, min(self.omega, self.omega_max))

    # --- Aeroprop model ---
    def get_thrust(self) -> float:
        """Return thrust [N] via quadratic law T = T_max * (omega/omega_max)^2."""
        if self.omega_max <= 0.0:
            return 0.0
        ratio = self.omega / self.omega_max
        return self.T_max_N * (ratio * ratio)


class Deflector:
    def __init__(self):
        self.max_deflection = math.radians(25)       # [rad]
        self.max_rate       = math.radians(15)       # [rad/s]
        self.x = 0.0  # achieved deflection (pitch axis), rad
        self.y = 0.0  # achieved deflection (yaw axis),   rad

    def set_deflection(self, target_x, target_y, dt):
        """Track target angles with rate and magnitude limits; store achieved."""
        max_step = self.max_rate * dt

        # X axis
        dx = target_x - self.x
        dx = max(-max_step, min(max_step, dx))
        self.x += dx
        self.x = max(-self.max_deflection, min(self.max_deflection, self.x))

        # Y axis
        dy = target_y - self.y
        dy = max(-max_step, min(max_step, dy))
        self.y += dy
        self.y = max(-self.max_deflection, min(self.max_deflection, self.y))

    def get_deflection(self):
        return self.x, self.y  # radians


class DroneState:
    def __init__(self):
        self.position = np.zeros(3)       # [x, y, z]
        self.velocity = np.zeros(3)       # [vx, vy, vz]
        self.acceleration = np.zeros(3)   # [ax, ay, az]
        self.orientation = np.zeros(3)    # [pitch, yaw, roll] or use quaternions if needed
        self.angular_velocity = np.zeros(3)  # [pitch_rate, yaw_rate, roll_rate]

    def __repr__(self):
        return f"Pos: {self.position}, Vel: {self.velocity}, Acc: {self.acceleration}"


class FlightLog:
    def __init__(self):
        self.records = []

    def add_state(self, state: DroneState, timestamp: float):
        self.records.append({
            't': timestamp,
            'pos': state.position.copy(),
            'vel': state.velocity.copy(),
            'acc': state.acceleration.copy(),
            'ori': state.orientation.copy(),
            'rates': state.angular_velocity.copy(),
        })



class ControlSystem:
    def __init__(self, Kp=3.0, Ki=0.6, Kd=0.0, Kv=1.2, dt=0.01,
                 max_deflect=math.radians(15), i_clamp=None):
        self.Kp, self.Ki, self.Kd, self.Kv = Kp, Ki, Kd, Kv
        self.dt = dt
        self.max_deflect = max_deflect
        self.i_clamp = i_clamp if i_clamp is not None else 0.5 * max_deflect

        self.prev_error_pitch = 0.0
        self.prev_error_yaw   = 0.0
        self.integral_pitch   = 0.0
        self.integral_yaw     = 0.0

    @staticmethod
    def _sat(x, lim):
        return max(-lim, min(lim, x))

    def compute_target_deflection(self, drone, target_pitch, target_yaw):
        # assume radians
        pitch = drone.drone_state.orientation[0]
        yaw   = drone.drone_state.orientation[1]
        p_dot = drone.drone_state.angular_velocity[0]
        y_dot = drone.drone_state.angular_velocity[1]

        # --- pitch loop
        e_p = target_pitch - pitch
        self.integral_pitch += e_p * self.dt
        self.integral_pitch = self._sat(self.integral_pitch, self.i_clamp)
        d_p = (e_p - self.prev_error_pitch) / self.dt
        self.prev_error_pitch = e_p

        raw_pitch = (self.Kp * e_p +
                     self.Ki * self.integral_pitch +
                     self.Kd * d_p -        # set Kd=0 if Kv>0
                     self.Kv * p_dot)

        # --- yaw loop
        e_y = target_yaw - yaw
        self.integral_yaw += e_y * self.dt
        self.integral_yaw = self._sat(self.integral_yaw, self.i_clamp)
        d_y = (e_y - self.prev_error_yaw) / self.dt
        self.prev_error_yaw = e_y

        raw_yaw = (self.Kp * e_y +
                   self.Ki * self.integral_yaw +
                   self.Kd * d_y -
                   self.Kv * y_dot)

        # saturate outputs to physical limits
        cmd_pitch = self._sat(raw_pitch, self.max_deflect)
        cmd_yaw   = self._sat(raw_yaw,   self.max_deflect)

        return cmd_pitch, cmd_yaw

    def update(self, drone, target_pitch, target_yaw):
        cmd_pitch, cmd_yaw = self.compute_target_deflection(drone, target_pitch, target_yaw)
        drone.deflector.set_deflection(cmd_pitch, cmd_yaw, self.dt)


class Battery:
    BATTERY_PARAMETERS = {
        "SLS Quantum 3000mAh 3S1P 11,1V 30C/60C": {
            "voltage": 11.1, #V
            "configuration": "3S1P",
            "capacity": 3000, #mAh
            "continuous discharge": 90, # A, 30C, max
            "burst discharge": 180, # A, 60C, max,
            "charge": 15, # A, 5C, max
            "weight": 235 # g (including Cable and Plug)
        }}

    def __init__(self, model: str = "SLS Quantum 3000mAh 3S1P 11,1V 30C/60C"):
        self.voltage = self.BATTERY_PARAMETERS[model]["voltage"]
        self.capacity = self.BATTERY_PARAMETERS[model]['capacity']
        self.weight = self.BATTERY_PARAMETERS[model]['weight']
        self.continuous_discharge = self.BATTERY_PARAMETERS[model]['continuous discharge']
        self.burst_discharge = self.BATTERY_PARAMETERS[model]['burst discharge']