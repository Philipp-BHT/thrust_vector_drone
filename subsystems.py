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
    def __init__(self, max_deflection=math.radians(20), max_rate=math.radians(60)):
        self.max_deflection = max_deflection
        self.max_rate = max_rate
        self._alpha = 0.0
        self._beta  = 0.0

    def get_deflection(self):
        return self._alpha, self._beta

    def set_deflection(self, alpha_target, beta_target, dt):
        # slew _alpha toward alpha_target by max_rate*dt
        step = self.max_rate * dt
        def slew(curr, target):
            delta = target - curr
            if abs(delta) <= step: return target
            return curr + math.copysign(step, delta)
        # clamp targets (safety)
        alpha_target = max(-self.max_deflection, min(self.max_deflection, alpha_target))
        beta_target  = max(-self.max_deflection, min(self.max_deflection,  beta_target))
        # integrate
        self._alpha = slew(self._alpha, alpha_target)
        self._beta  = slew(self._beta,  beta_target)



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


class AltitudeController:
    def __init__(self, control_params = None, dt=0.01,
                 i_clamp=3.0,  # clamp on the integral term (m/s)
                 min_cos=0.2,  # avoid divide-by-near-zero when tilted hard
                 a_cmd_limit=5.0):  # vertical accel command limit (m/s^2)
        self.Kp, self.Ki, self.Kv = [control_params.Kp, control_params.Ki, control_params.Kv] if (
            control_params) else [1.2, 0.4, 0.8]
        self.dt = dt
        self.i = 0.0
        self.i_clamp = i_clamp
        self.min_cos = min_cos
        self.a_cmd_limit = a_cmd_limit

    @staticmethod
    def _sat(x, lim): return max(-lim, min(lim, x))

    def step(self, drone, z_ref):
        """Return throttle u in [0..1] to hold altitude z_ref (meters)."""
        # --- measurements ---
        z = drone.drone_state.position[2]  # world Z up
        vz = drone.drone_state.velocity[2]  # world vertical speed

        # --- error ---
        e_z = z_ref - z

        # --- PI + rate damping on vertical speed (vz_ref = 0) ---
        self.i = self._sat(self.i + e_z * self.dt, self.i_clamp)
        a_cmd = self.Kp * e_z + self.Ki * self.i - self.Kv * vz  # desired vertical accel (m/s^2)
        a_cmd = self._sat(a_cmd, self.a_cmd_limit)

        # --- thrust direction & tilt compensation ---
        # use the CURRENT deflector to know where thrust points
        alpha, beta = drone.deflector.get_deflection()
        t_hat_B = drone._thrust_direction_body(alpha, beta)
        R_BW = drone._q_to_R_BW(drone._q)
        dir_W = R_BW @ t_hat_B
        dir_W_z = max(dir_W[2], self.min_cos)  # tilt compensation

        # --- required thrust magnitude ---
        m, g = drone.mass, drone.g
        # Want: (T*dir_W_z - m*g)/m  ≈  a_cmd   ->   T = m*(g + a_cmd)/dir_W_z
        T_req = m * (g + a_cmd) / dir_W_z

        # --- convert to throttle u (T ≈ T_max * u^2) ---
        T_max = drone.motor.T_max_N
        T_req = max(0.0, min(T_req, T_max))  # saturate to physical limit
        u = (T_req / T_max) ** 0.5

        # --- simple anti-windup: only integrate when not hard-saturated outward ---
        saturated = (T_req <= 1e-6) or (T_req >= T_max - 1e-6)
        if saturated:
            # If pushing further in the same direction, pause integral growth
            pass  # (already handled by clamping; keep it simple for now)

        return u


class OrientationController:
    def __init__(self,
                 control_params=None,
                 dt=0.01,
                 max_deflect=math.radians(15),
                 i_clamp=None):
        self.Kp, self.Ki, self.Kv = [control_params.Kp, control_params.Ki, control_params.Kv] if (
            control_params) else [2.5, 0, 6]

        self.dt = dt
        self.max_deflect = max_deflect
        self.i_clamp = 0.5*max_deflect if i_clamp is None else i_clamp

        self.i_pitch = 0
        self.i_yaw   = 0

    @staticmethod
    def _sat(x, lim): return max(-lim, min(lim, x))

    def compute_target_deflection(self, drone, ref_pitch=0.0, ref_yaw=0.0):
        pitch, yaw = drone.drone_state.orientation[0], drone.drone_state.orientation[1]
        p_dot, y_dot = drone.drone_state.angular_velocity[0], drone.drone_state.angular_velocity[1]
        e_p, e_y = (ref_pitch - pitch), (ref_yaw - yaw)

        u_p = self.Kp * e_p + self.Ki * self.i_pitch - self.Kv * p_dot
        u_y = self.Kp * e_y + self.Ki * self.i_yaw - self.Kv * y_dot
        return -u_p, -u_y

class PositionController:
    def __init__(self,
                 control_params=None, dt=0.01,
                 i_clamp=2.0,
                 max_tilt_deg=8.0,
                 sign_pitch=1.0,
                 sign_roll =-1.0):
        self.Kp, self.Ki, self.Kv = [control_params.Kp, control_params.Ki, control_params.Kv] if (
            control_params) else [0.1, 0, 1]
        self.dt = dt
        self.ix = 0.0
        self.iy = 0.0
        self.i_clamp = i_clamp
        self.max_tilt = math.radians(max_tilt_deg)
        self.sign_pitch = sign_pitch
        self.sign_roll  = sign_roll
        self.x_ref = None  # will be captured on first hold
        self.y_ref = None

    @staticmethod
    def _sat(x, lim): return max(-lim, min(lim, x))

    def capture_here(self, drone):
        x, y, _ = drone.drone_state.position
        self.x_ref, self.y_ref = float(x), float(y)
        self.ix = 0.0
        self.iy = 0.0
        print("Capture at ", self.x_ref, self.y_ref)
        print(drone.drone_state.position)

    def capture_ahead_tau(self, drone, tau=1):
        px, py, _ = drone.drone_state.position
        vx, vy, _ = drone.drone_state.velocity
        ax, ay, _ = drone.drone_state.acceleration
        self.x_ref = px + vx * tau
        self.y_ref = py + vy * tau

    def capture_ahead_stop(self, drone, max_tilt_deg=10.0):
        px, py, _ = drone.drone_state.position
        vx, vy, _ = drone.drone_state.velocity
        v = math.hypot(vx, vy)
        if v < 1e-3:
            self.x_ref, self.y_ref = px, py
            return

        g = drone.g
        a_max = g * math.tan(math.radians(max_tilt_deg))
        s_over_v = v / (2.0 * a_max)  # = s / ||v||
        x_ref = px + s_over_v * vx
        y_ref = py + s_over_v * vy
        self.x_ref, self.y_ref = x_ref, y_ref

    def step(self, drone):
        """Return (pitch_ref, roll_ref) in radians to hold XY."""
        assert self.x_ref is not None and self.y_ref is not None, "Call capture_here() first."
        x, y, _  = drone.drone_state.position

        # print("Desired Position: ", round(self.x_ref, 2), " ", round(self.y_ref, 2), ", Actual position: ", round(drone.drone_state.position[0], 2), " ", round(drone.drone_state.position[1], 2))
        vx, vy, _ = drone.drone_state.velocity
        ex = self.x_ref - x
        ey = self.y_ref - y

        self.ix = self._sat(self.ix + ex * self.dt, self.i_clamp)
        self.iy = self._sat(self.iy + ey * self.dt, self.i_clamp)
        ax_cmd = self.Kp*ex + self.Ki*self.ix - self.Kv*vx
        ay_cmd = self.Kp*ey + self.Ki*self.iy - self.Kv*vy

        g = drone.g
        pitch_ref = self.sign_pitch * self._sat(ax_cmd / g, self.max_tilt)
        roll_ref  = self.sign_roll  * self._sat(ay_cmd / g, self.max_tilt)

        return pitch_ref, roll_ref



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