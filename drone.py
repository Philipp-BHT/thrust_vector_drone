from subsystems import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class DroneSim:
    def __init__(self, weight, diameter, height, motor_model: str = None):
        self.weight = weight
        self.diameter = diameter
        self.height = height

        self.motor = Motor(motor_model) if motor_model else Motor()
        self.deflector = Deflector()
        self.drone_state = DroneState()
        self.flight_records = FlightLog()
        self.orientation_control = OrientationController()
        self.altitude_control = AltitudeController()
        self.position_control = PositionController()
        self.battery = Battery()

        self.g = 9.81
        self.mass = self.weight / self.g
        self.thrust_N = self.weight

        self.lever = 0.04  # [m] nozzle offset below CoM (tune 0.04–0.12)
        self.J = np.diag([0.2, 0.2, 0.4])  # [kg m^2] inertia about body X,Y,Z (tune)
        self.c_omega = 0.2  # [N m s/rad] rot. damping (tune)

        # attitude state helper (quaternion)
        self._q = np.array([1.0, 0.0, 0.0, 0.0])  # w,x,y,z

        self.last_T = 0
        self.last_tW = 0

    @staticmethod
    def _thrust_direction_body(x_defl, y_defl):
        """
        Map small deflections (pitch=x, yaw=y) to a unit thrust vector in body frame.
        Convention: nominal thrust along -Z_B; small-angle approx.
        """
        t = np.array([ y_defl, -x_defl, 1.0 ])
        return t / np.linalg.norm(t)

    def _omega_quat(self, w):  # w = [p,q,r]
        return np.array([0.0, w[0], w[1], w[2]])

    def _q_mul(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ])

    def _q_to_R_BW(self, q):
        # body -> world
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])

    def _q_to_euler_zyx(self, q):
        # returns pitch (Y), yaw (Z), roll (X) in radians to match your state ordering
        w, x, y, z = q
        # ZYX
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        sp = -2 * (x * z - w * y)
        sp = +1.0 if sp > +1.0 else (-1.0 if sp < -1.0 else sp)
        pitch = math.asin(sp)
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        return np.array([roll, pitch, yaw])

    def update(self, dt, throttle,
               manual_deflection=None,  # (alpha, beta) in radians if mode="manual"
               ref_pitch=0.0, ref_yaw=0.0,  # desired attitude if closed-loop
               t_now=0.0):

        # choose ONE deflection source
        if manual_deflection:
            alpha = manual_deflection[0]
            beta = manual_deflection[1]
        else:
            alpha, beta = self.orientation_control.compute_target_deflection(self, ref_pitch, ref_yaw)
        # alpha, beta = manual_deflection
        # print(alpha, beta)

        self.deflector.set_deflection(alpha, beta, dt)

        # motor & dynamics
        self.motor.set_throttle(throttle)
        self.motor.update(dt)
        T = self.motor.get_thrust()

        x_defl, y_defl = self.deflector.get_deflection()
        t_hat_B = self._thrust_direction_body(x_defl, y_defl)
        f_B = T * t_hat_B

        # --- Forces (world) as you already have ---
        R_WB = self._q_to_R_BW(self._q)  # world->body from current q
        # R_BW = R_WB.T  # body->world
        f_thrust_W = R_WB @ f_B
        # before computing a_W:
        c_lin = 2 * self.mass  # N per (m/s) – tune 0.3..1.0 * mass
        f_W = f_thrust_W + np.array([0, 0, -self.mass * self.g]) - c_lin * self.drone_state.velocity
        a_W = f_W / self.mass

        self.drone_state.acceleration = a_W
        # ground contact (simple clamp)
        if self.drone_state.position[2] <= 0 and a_W[2] < 0:
            a_W[2] = 0.0
            self.drone_state.velocity[2] = 0.0
        self.drone_state.velocity += a_W * dt
        self.drone_state.position += self.drone_state.velocity * dt
        if self.drone_state.position[2] < 0:
            self.drone_state.position[2] = 0.0
            self.drone_state.velocity[2] = max(0.0, self.drone_state.velocity[2])

        # --- ROTATIONAL DYNAMICS ---
        # Lever arm: nozzle below CoM along -Z_B
        r_T = np.array([0.0, 0.0, -self.lever])  # [m] in body frame
        tau_B = np.cross(r_T, f_B)  # [Nm] pitch/yaw from lateral thrust
        # simple rot. damping
        tau_B += -self.c_omega * self.drone_state.angular_velocity

        J = self.J
        w = self.drone_state.angular_velocity
        w_dot = np.linalg.inv(J) @ (tau_B - np.cross(w, J @ w))
        self.drone_state.angular_velocity = w + w_dot * dt

        q_dot = 0.5 * self._q_mul(self._q, self._omega_quat(self.drone_state.angular_velocity))
        self._q = self._q + q_dot * dt
        self._q /= np.linalg.norm(self._q)  # normalize

        self.drone_state.orientation = self._q_to_euler_zyx(self._q)

        self.last_T = T
        self.last_tW = f_thrust_W / max(T, 1e-9)

        self.flight_records.add_state(self.drone_state, t_now)


if __name__ == "__main__":
    t=0
    z_ref = 0 # meters
    T_end = 40.0
    dt = 0.01
    steps = []
    input_signal = []

    drone = DroneSim(weight=1.9 * 9.81,
                     diameter=0.2,
                     height=0.3,
                     motor_model="T-Motor F80 PRO 2408 Brushless Motor")

    while t < T_end:
        if t > 5.0:
            z_ref = 2
        alpha_cmd, beta_cmd = drone.orientation_control.compute_target_deflection(
            drone, ref_pitch=0.0, ref_yaw=0.0)
        throttle = drone.altitude_control.step(drone, z_ref)
        drone.update(dt,
                     throttle,
                     manual_deflection=(alpha_cmd, beta_cmd),
                     ref_pitch=0.0,
                     ref_yaw=0.0,
                     t_now=t)
        t += dt
        steps.append(t)
        input_signal.append(z_ref)


    z_pos = [float(pos["pos"][2]) for pos in drone.flight_records.records]
    print("Z_Positions: ", z_pos)
    print("t:", steps)
    print("input: ", input_signal)
    fig, ax = plt.subplots()
    ax.plot(steps, z_pos)
    ax.plot(steps, input_signal)
    plt.show()