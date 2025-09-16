import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
from drone import DroneSim, ControlParams


def draw_reference_rings(center, radius=0.2, segments=100):
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
    glRotatef(-math.degrees(yaw),   1, 0, 0)  # yaw about Z
    glRotatef(-math.degrees(pitch), 0, 1, 0)  # pitch about Y
    glRotatef(-math.degrees(roll),  0, 0, 1)  # roll about X

    # --- Body geometry (cylinder) ---
    quadric = gluNewQuadric()
    glColor3f(0.3, 0.3, 0.5)
    radius = drone.diameter * 0.5
    gluCylinder(quadric, radius, radius, drone.height, 20, 20)

    # Nozzle (local to body). alpha = about Y, beta = about X:
    glPushMatrix()
    alpha_rad, beta_rad = drone.deflector.get_deflection()
    glRotatef(math.degrees(beta_rad), 1, 0, 0)  # roll axis (X)
    glRotatef(math.degrees(alpha_rad), 0, 1, 0)  # pitch axis (Y)

    glTranslatef(0, 0, -0.15)
    glColor3f(0.8, 0.3, 0.5)
    gluCylinder(quadric, radius*0.75, radius, drone.height * 0.5, 10, 10)
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
    spacing = 1  # Distance between grid lines

    # Draw vertical and horizontal lines
    for i in range(-grid_size, grid_size + 1, spacing):
        glVertex3f(i, -1, -grid_size)  # Move slightly up (-1 instead of -5)
        glVertex3f(i, -1, grid_size)

        glVertex3f(-grid_size, -1, i)
        glVertex3f(grid_size, -1, i)

    glEnd()
    glPopMatrix()


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
    draw_text(font, f"Speed: {round(sum(drone.drone_state.velocity),2)} m/s", screen_width - 10, screen_height - 20, align="right")
    draw_text(font, f"Altitude {round(drone.drone_state.position[2],2)} m", screen_width - 10, screen_height - 40, align="right")

def draw_point_3d(drone, obj, color=(1.0, 0.2, 0.2, 1.0)):
    if any([drone.position_control.x_ref, drone.position_control.y_ref]):
        glPushMatrix()
        glTranslatef(drone.position_control.x_ref, drone.position_control.y_ref, 2)
        glColor4f(*color)  # with GL_COLOR_MATERIAL enabled in your setup
        gluSphere(obj, 0.1, 16, 12)
        glPopMatrix()

def display(font, screen_width, screen_height, drone, camera_angle_yaw, camera_angle_pitch, stars):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    draw_background()

    # ---- Camera for this frame ----
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    SPHERE_QUAD = gluNewQuadric()
    gluQuadricNormals(SPHERE_QUAD, GLU_SMOOTH)

    center_of_mass = 0.0
    x_position, y_position, z_position = drone.drone_state.position

    camera_distance = 3.0   # meters; change freely now
    camera_height   = 1.5   # meters

    yaw_camera_angle_rad   = np.deg2rad(camera_angle_yaw)
    pitch_camera_angle_rad = np.deg2rad(camera_angle_pitch)

    eye_x = x_position + camera_distance * math.cos(yaw_camera_angle_rad)
    eye_y = y_position + camera_distance * math.sin(yaw_camera_angle_rad)
    eye_z = z_position + camera_height * math.sin(pitch_camera_angle_rad)

    gluLookAt(eye_x, eye_y, eye_z,
              x_position, y_position, z_position + center_of_mass,
              0, 0, 1)

    glPushMatrix()
    draw_grid()
    draw_drone(drone)
    draw_point_3d(drone, SPHERE_QUAD)
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

    # ---- One-time GL setup ----
    glViewport(0, 0, screen_width, screen_height)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    # Tighter FOV and meter-friendly near/far planes
    gluPerspective(40.0, screen_width / screen_height, 0.01, 50.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()           # IMPORTANT: start modelview clean (no global translate!)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    font = pygame.font.SysFont("Arial", 18)
    clock = pygame.time.Clock()

    drone = (DroneSim(weight=1.9 * 9.81,
                      diameter=0.2,
                      height=0.3,
                      motor_model="T-Motor F80 PRO 2408 Brushless Motor",
                      altitude_control=ControlParams(Kp=2, Kv=0.5, Ki=0),
                      position_control=ControlParams(Kp=0.7, Kv=0.2, Ki=0),
                      orientation_control=ControlParams(Kp=2, Kv=1.2, Ki=0),
                      ))

    camera_angle_pitch = 0
    camera_angle_yaw = 0
    launched = False
    running = True
    manual_prev = False
    z_ref = 2.0

    t = 0.0
    stars = [(np.random.uniform(-500, 500), np.random.uniform(100, 500), np.random.uniform(-500, 500))
             for _ in range(200)]

    while running:
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)

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

        if launched:
            throttle = drone.altitude_control.step(drone, z_ref)
        else:
            throttle = 0
        alpha, beta = drone.deflector.get_deflection()
        step = drone.deflector.max_rate * dt

        if keys[K_a]: alpha -= step
        if keys[K_d]: alpha += step
        if keys[K_s]: beta -= step
        if keys[K_w]: beta += step

        alpha = max(-drone.deflector.max_deflection, min(drone.deflector.max_deflection, alpha))
        beta = max(-drone.deflector.max_deflection, min(drone.deflector.max_deflection, beta))

        pressed = keys[K_a] or keys[K_d] or keys[K_s] or keys[K_w]
        manual_deflection = (alpha, beta) if pressed else None

        if manual_prev and not pressed:
            drone.position_control.capture_here(drone)

        if drone.position_control.x_ref is None:
            drone.position_control.capture_here(drone)

        if not pressed:
            pitch_ref, roll_ref = drone.position_control.step(drone)
        else:
            pitch_ref, roll_ref = 0, 0

        drone.update(dt, throttle, manual_deflection=manual_deflection, ref_pitch=roll_ref, ref_yaw=pitch_ref)

        drone.flight_records.add_state(drone.drone_state, t)
        t += dt
        manual_prev = pressed

        display(font, screen_width, screen_height, drone, camera_angle_yaw, camera_angle_pitch, stars)

    pygame.quit()


if __name__ == "__main__":
    run_pygame_simulation()


