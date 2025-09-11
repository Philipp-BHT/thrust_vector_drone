import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
from drone import DroneSim


def draw_reference_rings(center, radius=15, segments=100):
    """ Draws 3 axis-aligned rings (X, Y, Z) centered at the given position. """
    x, y, z = center
    glLineWidth(1)

    # Circle in XY plane (Z-axis normal)
    glColor3f(1, 0, 0)  # Red
    glBegin(GL_LINE_LOOP)
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        glVertex3f(x + radius * math.cos(theta), y + radius * math.sin(theta), z)
    glEnd()

    # Circle in XZ plane (Y-axis normal)
    glColor3f(0, 1, 0)  # Green
    glBegin(GL_LINE_LOOP)
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        glVertex3f(x + radius * math.cos(theta), y, z + radius * math.sin(theta))
    glEnd()

    # Circle in YZ plane (X-axis normal)
    glColor3f(0, 0.5, 1)  # Blue
    glBegin(GL_LINE_LOOP)
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        glVertex3f(x, y + radius * math.cos(theta), z + radius * math.sin(theta))
    glEnd()


def draw_drone(drone: DroneSim):
    glPushMatrix()

    # --- World-space helpers ---
    x, y, z = drone.drone_state.position
    vx, vy, vz = drone.drone_state.velocity

    draw_reference_rings((x, y, z))

    # Velocity vector (green)
    glColor3f(0, 1, 0)
    glBegin(GL_LINES)
    glVertex3f(x, y, z)
    glVertex3f(x + vx*3.0, y + vy*3.0, z + vz*3.0)
    glEnd()

    # Thrust vector in WORLD (red) using cached values from update()
    if hasattr(drone, "last_tW") and hasattr(drone, "last_T"):
        Tx, Ty, Tz = drone.last_tW * drone.last_T
        glColor3f(1, 0, 0)
        glBegin(GL_LINES)
        glVertex3f(x, y, z)
        glVertex3f(x + Tx*0.01, y + Ty*0.01, z + Tz*0.01)  # scale for viz
        glEnd()

    # --- Move to body frame and orient the model ---
    glTranslatef(x, y, z)

    pitch, yaw, roll = drone.drone_state.orientation  # RAD
    # Apply Rz(-yaw) → Ry(-pitch) → Rx(-roll); degrees for GL
    glRotatef(-math.degrees(yaw),   0, 0, 1)  # yaw about Z
    glRotatef(-math.degrees(pitch), 0, 1, 0)  # pitch about Y
    glRotatef(-math.degrees(roll),  1, 0, 0)  # roll about X

    # --- Body geometry (cylinder) ---
    quadric = gluNewQuadric()
    glColor3f(0, 0, 0)
    gluCylinder(quadric, drone.diameter*40, drone.diameter*40, drone.height*40, 20, 20)

    # --- Nozzle/deflector (in body frame) ---
    glPushMatrix()
    alpha_rad, beta_rad = drone.deflector.get_deflection()  # RAD
    glRotatef(math.degrees(beta_rad),  1, 0, 0)  # yaw β about Z
    glRotatef(math.degrees(alpha_rad), 0, 1, 0)  # pitch α about Y
    glTranslatef(0, 0, -5)                        # put it under the body
    glColor3f(0.5, 0.7, 1)
    gluCylinder(quadric, 4, 2, 10, 10, 10)
    glPopMatrix()

    glPopMatrix()

    # --- Trajectory points (read positions from your log) ---
    trajectory = [rec['state'].position for rec in drone.flight_records.records] \
                 if (drone.flight_records.records and 'state' in drone.flight_records.records[0]) \
                 else [rec['pos'] for rec in drone.flight_records.records]  # support either log format
    glPointSize(4)
    glBegin(GL_POINTS)
    for i, (px, py, pz) in enumerate(trajectory[-10000:]):  # cap for perf
        glColor4f(1, 1, 1, (i+1)/max(1, len(trajectory[-10000:])))
        glVertex3f(px, py, pz)
    glEnd()


def draw_grid():
    """ Draws a large ground grid for reference. """
    glPushMatrix()
    glRotatef(90, 1, 0, 0)  # Rotate around local X-axis (Pitch)

    glColor3f(0.3, 0.3, 0.3)  # Light grey
    glBegin(GL_LINES)

    # Grid size and spacing
    grid_size = 10000  # Increase to make it visible
    spacing = 5  # Distance between grid lines

    # Draw vertical and horizontal lines
    for i in range(-grid_size, grid_size + 1, spacing):
        glVertex3f(i, -1, -grid_size)  # Move slightly up (-1 instead of -5)
        glVertex3f(i, -1, grid_size)

        glVertex3f(-grid_size, -1, i)
        glVertex3f(grid_size, -1, i)

    glEnd()
    glPopMatrix()


def draw_stars(stars):
    """ Draws stars in the background for reference. """
    glPointSize(2)
    glBegin(GL_POINTS)
    glColor3f(1, 1, 1)
    for x, y, z in stars:
        glVertex3f(x, y, z)
    glEnd()


def draw_background():
    """ Draws a full-screen gradient background. """
    glMatrixMode(GL_PROJECTION)  # Switch to projection mode
    glPushMatrix()
    glLoadIdentity()
    glOrtho(-1, 1, -1, 1, -1, 1)  # Set orthographic projection

    glMatrixMode(GL_MODELVIEW)  # Switch back to model view
    glPushMatrix()
    glLoadIdentity()

    glDisable(GL_DEPTH_TEST)  # Ensure it's always behind everything
    glBegin(GL_QUADS)

    # Top color (Bright Blue Sky)
    glColor3f(0.1, 0.3, 0.8)
    glVertex2f(-1, 1)
    glVertex2f(1, 1)

    # Bottom color (Dark Space)
    glColor3f(0, 0, 0.2)
    glVertex2f(1, -1)
    glVertex2f(-1, -1)

    glEnd()
    glEnable(GL_DEPTH_TEST)  # Restore depth testing

    # Restore previous projection
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)  # Back to model view mode


def draw_text(font, text, x, y, align="left", color=(255, 255, 255)):
    """Draw text using Pygame on top of OpenGL with alignment options."""
    text_surface = font.render(text, True, color)
    text_data = pygame.image.tostring(text_surface, "RGBA", True)

    text_width = text_surface.get_width()

    if align == "right":
        x -= text_width
    elif align == "center":
        x -= text_width // 2

    glWindowPos2d(x, y)
    glDrawPixels(text_surface.get_width(), text_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)


def draw_controls(font, screen_height):
    draw_text(font, "Mouse: Move camera", 10, screen_height - 20)
    draw_text(font, "W/S/A/D: Thrust vectoring", 10, screen_height - 40)
    draw_text(font, "Space: Launch", 10, screen_height - 60)


def draw_stats(font, screen_width, screen_height, drone):
    draw_text(font, f"Speed: {int(sum(drone.drone_state.velocity))} m/s", screen_width - 10, screen_height - 20, align="right")
    draw_text(font, f"Altitude {int(drone.drone_state.position[2])} m", screen_width - 10, screen_height - 40, align="right")


def display(font, screen_width, screen_height, drone, camera_angle_yaw, camera_angle_pitch, stars):
    """ Updates the camera to follow the rocket. """
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    draw_background()

    glPushMatrix()

    center_of_mass = 0
    x_position, y_position, z_position = drone.drone_state.position

    height_factor = min(250, abs(z_position) / 5)  # Increase distance at high altitude

    camera_distance = 5 # + height_factor
    camera_height = 15 # + height_factor



    yaw_camera_angle_rad = np.deg2rad(camera_angle_yaw)
    pitch_camera_angle_rad = np.deg2rad(camera_angle_pitch)

    eye_x = x_position + camera_distance * math.cos(yaw_camera_angle_rad)
    eye_y = y_position + camera_distance * math.sin(yaw_camera_angle_rad)
    eye_z = z_position + camera_height * math.sin(pitch_camera_angle_rad)

    gluLookAt(eye_x, eye_y, eye_z,
              x_position, y_position, z_position + center_of_mass,
              0, 0, 1)

    draw_stars(stars)
    draw_grid()
    draw_drone(drone)
    draw_controls(font, screen_height)
    draw_stats(font, screen_width, screen_height, drone)

    glPopMatrix()
    pygame.display.flip()

def get_current_waypoint(t, waypoints):
    # Find the last keyframe ≤ current time
    angles = sorted(waypoints.keys())
    for i in reversed(angles):
        if t >= i:
            return waypoints[i]['yaw'], waypoints[i]['pitch']
    return waypoints[angles[0]]['yaw'], waypoints[angles[0]]['pitch']   # Default to the first key if none matched


def run_pygame_simulation():
    pygame.init()
    screen_width, screen_height = 800, 600
    pygame.display.set_caption("Real-Time 3D Drone Simulation")
    screen = pygame.display.set_mode((screen_width, screen_height), DOUBLEBUF | OPENGL)

    glEnable(GL_DEPTH_TEST)
    gluPerspective(60, screen_width / screen_height, 0.1, 1000.0)
    glTranslatef(0, -5, -100)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    font = pygame.font.SysFont("Arial", 18)

    clock = pygame.time.Clock()

    drone = DroneSim(weight=1.9*9.81,
                     diameter=0.2,
                     height=0.4,
                     motor_model="T-Motor F80 PRO 2408 Brushless Motor")

    max_gimbal_rate = 20  # deg/s
    camera_angle_pitch = 0
    camera_angle_yaw = 0
    launched = False
    running = True

    t = 0.0
    stars = [(np.random.uniform(-500, 500), np.random.uniform(100, 500), np.random.uniform(-500, 500))
             for _ in range(200)]

    while running:
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)

        # stop when below ground

        if drone.drone_state.position[2] < 0:
            pygame.event.set_grab(False)
            pygame.mouse.set_visible(True)
            break

        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        mx, my = pygame.mouse.get_rel()
        camera_angle_pitch += my / 10
        camera_angle_yaw   -= mx / 10

        keys = pygame.key.get_pressed()
        if keys[K_SPACE]:
            launched = True

        # ---------- Controls ----------
        # Hover throttle (T ∝ u^2)
        u_hover = math.sqrt((drone.mass * drone.g) / drone.motor.T_max_N)
        throttle = 1 if launched else 0.0
        throttle = max(0.0, min(1.0, throttle))

        # Gimbal manual control in radians
        alpha, beta = drone.deflector.get_deflection()  # rad
        step = drone.deflector.max_rate * dt
        if keys[K_a]: alpha -= step
        if keys[K_d]: alpha += step
        if keys[K_s]: beta  -= step
        if keys[K_w]: beta  += step
        # spring-back when no key pressed
        ret = step * 0.5
        if not (keys[K_a] or keys[K_d]):
            alpha = alpha - ret if alpha > 0 else alpha + ret if alpha < 0 else 0.0
        if not (keys[K_w] or keys[K_s]):
            beta  = beta  - ret if beta  > 0 else beta  + ret if beta  < 0 else 0.0

        drone.deflector.set_deflection(alpha, beta, dt)

        # ---------- Physics step (always) ----------
        # Pass throttle ∈ [0..1]; targets = 0 to keep attitude neutral if you’re not using the attitude loop
        drone.update(dt=dt, throttle=throttle, target_pitch=None, target_yaw=None, t_now=t)
        drone.flight_records.add_state(drone.drone_state, t)
        t += dt

        display(font, screen_width, screen_height, drone, camera_angle_yaw, camera_angle_pitch, stars)

    pygame.quit()


if __name__ == "__main__":
    run_pygame_simulation()


