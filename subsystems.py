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
        self.max_deflection = math.radians(20)       # [rad]
        self.max_rate       = math.radians(80)       # [rad/s]
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
    def __init__(self,
                 Kp=3.0, Ki=0.4, Kv=1.6,          # PI + rate damping (Kd=0)
                 dt=0.01,
                 max_deflect=math.radians(20),    # match Deflector
                 i_clamp=None,
                 max_tilt_deg=10.0,               # HARD cap (deg)
                 soft_tilt_deg=7.0,               # start pushback (deg)
                 Kguard=8.0):                     # guard strength
        self.Kp, self.Ki, self.Kv = Kp, Ki, Kv
        self.dt = dt
        self.max_deflect = max_deflect
        self.i_clamp = 0.5*max_deflect if i_clamp is None else i_clamp

        self.max_tilt  = math.radians(max_tilt_deg)
        self.soft_tilt = math.radians(soft_tilt_deg)
        self.Kguard    = Kguard

        self.i_pitch = 0.0
        self.i_yaw   = 0.0

    @staticmethod
    def _sat(x, lim): return max(-lim, min(lim, x))

    def _guard_term(self, angle):
        """Zero inside soft_tilt; grows smoothly beyond to push back toward zero."""
        a = abs(angle)
        if a <= self.soft_tilt:
            return 0.0
        denom = max(self.max_tilt - self.soft_tilt, 1e-6)
        r = (a - self.soft_tilt) / denom            # 0 at soft, 1 at hard
        g = self.Kguard * math.tanh(2.0 * r)        # smooth, bounded
        return -g * math.copysign(1.0, angle)

    def compute_target_deflection(self, drone):
        pitch, yaw = drone.drone_state.orientation[0], drone.drone_state.orientation[1]
        p_dot, y_dot = drone.drone_state.angular_velocity[0], drone.drone_state.angular_velocity[1]

        # errors to level (0 target)
        e_p, e_y = -pitch, -yaw

        # PI with clamp
        self.i_pitch = self._sat(self.i_pitch + e_p * self.dt, self.i_clamp)
        self.i_yaw   = self._sat(self.i_yaw   + e_y * self.dt, self.i_clamp)

        u_p = self.Kp*e_p + self.Ki*self.i_pitch - self.Kv*p_dot
        u_y = self.Kp*e_y + self.Ki*self.i_yaw   - self.Kv*y_dot

        # soft tilt guard
        u_p += self._guard_term(pitch)
        u_y += self._guard_term(yaw)

        # hard guard
        if abs(pitch) >= self.max_tilt:
            u_p = -math.copysign(self.max_deflect, pitch)
            self.i_pitch *= 0.9   # bleed integrator a bit
        if abs(yaw)   >= self.max_tilt:
            u_y = -math.copysign(self.max_deflect, yaw)
            self.i_yaw   *= 0.9

        return self._sat(u_p, self.max_deflect), self._sat(u_y, self.max_deflect)

    def update(self, drone):
        cmd_pitch, cmd_yaw = self.compute_target_deflection(drone)
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