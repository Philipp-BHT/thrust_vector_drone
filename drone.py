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
        self.control_system = ControlSystem()
        self.battery = Battery()

        self.g = 9.81
        self.mass = self.weight / self.g
        self.thrust_N = self.weight

        self.lever = 0.04  # [m] nozzle offset below CoM (tune 0.04–0.12)
        self.J = np.diag([0.02, 0.02, 0.04])  # [kg m^2] inertia about body X,Y,Z (tune)
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

    @staticmethod
    def euler_to_R_world_from_body(pitch, yaw, roll):
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cr, sr = math.cos(roll), math.sin(roll)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        R_W_to_B = Rz @ Ry @ Rx
        return R_W_to_B

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

    def _q_to_R_WB(self, q):
        # world->body from quaternion (right-handed, unit q)
        w, x, y, z = q
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
        ])
        return R

    def _q_to_euler_zyx(self, q):
        # returns pitch (Y), yaw (Z), roll (X) in radians to match your state ordering
        w, x, y, z = q
        # ZYX
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        sp = -2 * (x * z - w * y)
        sp = +1.0 if sp > +1.0 else (-1.0 if sp < -1.0 else sp)
        pitch = math.asin(sp)
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        return np.array([pitch, yaw, roll])

    def update(self, dt, throttle, target_pitch=None, target_yaw=None, t_now=0.0):

        if target_pitch is not None and target_yaw is not None:
            cmd_x, cmd_y = self.control_system.compute_target_deflection(self, target_pitch, target_yaw)
            self.deflector.set_deflection(cmd_x, cmd_y, dt)


        self.motor.set_throttle(throttle)  # throttle in [0..1]
        self.motor.update(dt)
        T = self.motor.get_thrust()

        x_defl, y_defl = self.deflector.get_deflection()
        t_hat_B = self._thrust_direction_body(x_defl, y_defl)
        f_B = T * t_hat_B

        # --- Forces (world) as you already have ---
        R_WB = self._q_to_R_WB(self._q)  # world->body from current q
        R_BW = R_WB.T  # body->world
        f_thrust_W = R_BW @ f_B
        f_W = f_thrust_W + np.array([0, 0, -self.mass * self.g])  # gravity DOWN
        # (optional) linear drag:
        # f_W += -c_v * self.drone_state.velocity

        # --- Translational integration (same as you have) ---
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

        # Euler's equation (diagonal J)
        J = self.J
        w = self.drone_state.angular_velocity
        w_dot = np.linalg.inv(J) @ (tau_B - np.cross(w, J @ w))
        self.drone_state.angular_velocity = w + w_dot * dt

        # Quaternion integration: q_dot = 0.5 * q ⊗ [0, w]
        q_dot = 0.5 * self._q_mul(self._q, self._omega_quat(self.drone_state.angular_velocity))
        self._q = self._q + q_dot * dt
        self._q /= np.linalg.norm(self._q)  # normalize

        # Update Euler angles for your renderer/state (ZYX)
        self.drone_state.orientation = self._q_to_euler_zyx(self._q)

        # --- cache for renderer (world thrust vector) ---
        self.last_T = T
        self.last_tW = f_thrust_W / max(T, 1e-9)

        # log (ideally log a copy)
        self.flight_records.add_state(self.drone_state, t_now)


def plot_trajectory_3d(flight_log, show_quivers=False, every_n=20):
    """3D path of the drone. Optionally draw sparse heading arrows."""
    if not flight_log.records:
        print("No data to plot.")
        return

    P = np.array([r['pos'] for r in flight_log.records])  # shape (N,3)
    X, Y, Z = P[:,0], P[:,1], P[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Path
    ax.plot(X, Y, Z, linewidth=2)

    # Start/end markers
    ax.scatter(X[0], Y[0], Z[0], s=50, marker='o', label='start')
    ax.scatter(X[-1], Y[-1], Z[-1], s=50, marker='^', label='end')

    # Optional: sparse orientation arrows (uses yaw only as heading demo)
    if show_quivers:
        ORI = np.array([r['ori'] for r in flight_log.records])
        yaw = ORI[::every_n, 1]  # assuming orientation = [pitch, yaw, roll]
        Xs, Ys, Zs = X[::every_n], Y[::every_n], Z[::every_n]
        u = np.cos(yaw); v = np.sin(yaw); w = np.zeros_like(u)
        ax.quiver(Xs, Ys, Zs, u, v, w, length=0.2, normalize=True)

    # Equal aspect ratio
    _set_equal_3d(ax, X, Y, Z)

    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
    ax.legend(loc='upper left')
    ax.set_title('Drone trajectory (3D)')
    plt.tight_layout()
    plt.show()

def _set_equal_3d(ax, X, Y, Z):
    """Make 3D axes have equal scale."""
    x_range = X.max() - X.min()
    y_range = Y.max() - Y.min()
    z_range = Z.max() - Z.min()
    max_range = max(x_range, y_range, z_range)
    x_mid = (X.max() + X.min())/2
    y_mid = (Y.max() + Y.min())/2
    z_mid = (Z.max() + Z.min())/2
    ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
    ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
    ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)
    try:
        ax.set_box_aspect((1,1,1))
    except Exception:
        pass

def animate_trajectory_3d(flight_log):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    line, = ax.plot([], [], [], linewidth=2)
    start = ax.scatter([], [], [], s=50, marker='o')
    end   = ax.scatter([], [], [], s=50, marker='^')

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        return line, start, end

    def update(frame):
        P = np.array([r['pos'] for r in flight_log.records[:frame]])
        if P.size == 0:
            return line, start, end
        X, Y, Z = P[:,0], P[:,1], P[:,2]
        line.set_data(X, Y)
        line.set_3d_properties(Z)

        # Start/end markers
        start._offsets3d = (np.array([X[0]]), np.array([Y[0]]), np.array([Z[0]]))
        end._offsets3d   = (np.array([X[-1]]), np.array([Y[-1]]), np.array([Z[-1]]))

        _set_equal_3d(ax, X, Y, Z)
        ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
        return line, start, end

    ani = FuncAnimation(fig, update, frames=len(flight_log.records),
                        init_func=init, interval=30, blit=False, repeat=False)
    plt.show()



if __name__ == "__main__":
    dt = 0.01
    t_end = 5.0
    n = int(t_end / dt)

    drone = DroneSim(weight=2.0 * 9.81,
                     diameter=0.2,
                     height=0.4,
                     motor_model="T-Motor F80 PRO 2408 Brushless Motor")

    for i in range(n):
        t = i * dt
        throttle = 0.9
        target_pitch = 0.10  # rad
        target_yaw = 0.15  # rad
        drone.update(dt=dt, throttle=throttle, target_pitch=target_pitch, target_yaw=target_yaw, t_now=t)

    animate_trajectory_3d(drone.flight_records)
