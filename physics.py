from drone import DroneSim
import math
import numpy as np


class DronePhysicsModel:
    GRAVITY = 9.81  # m/s^2

    @staticmethod
    def calculate_center_of_mass(drone: DroneSim) -> float:
        """
        Computes the center of mass of a drone.

        Parameters:
            drone: (DroneSim): The drone object

        Returns:
            float: Center of mass (m).
        """

        # For now, mass is assumed to be evenly distributed
        center_of_mass = drone.weight * drone.height / 2

        return center_of_mass

    @staticmethod
    def get_air_density(altitude: float) -> float:
        """
        Returns air density (kg/m^3) based on altitude using an approximate exponential model.
        """

        if altitude < 11000:
            return 1.225 * math.exp(-altitude / 8000)
        elif altitude < 25000:
            return 0.36391 * math.exp(-(altitude - 11000) / 6000)
        elif altitude < 50000:
            return 0.08803 * math.exp(-(altitude - 25000) / 5500)
        else:
            return 0.00001846 * math.exp(-(altitude - 50000) / 8000)

    @staticmethod
    def calculate_thrust(drone: DroneSim) -> float:
        """
        Calculates the thrust of a drone.
        The thrust is dependent on the air density. The higher the drone, the lower the density, the lower the thrust.
        Parameters:
            drone: (DroneSim): The drone object.
        Returns:
        """

        air_density = DronePhysicsModel.get_air_density(drone.drone_state.position[2])
        air_entrance_velocity = 0
        air_exit_velocity = 0

        thrust = 1/2 * air_density * math.pi * drone.motor.propeller_radius**2 * (air_entrance_velocity**2 - air_exit_velocity**2)

        return thrust

    @staticmethod
    def get_torque(drone: DroneSim) -> tuple[float, float, float]:
        """Computes torque based on the difference between thrust vector and body orientation."""
        thrust_force = DronePhysicsModel.calculate_thrust(drone)
        lever_arm = drone.height - DronePhysicsModel.calculate_center_of_mass(drone)

        pitch_torque = lever_arm * thrust_force * math.sin(math.radians(drone.deflector.thrust_pitch_local))
        yaw_torque = lever_arm * thrust_force * math.sin(math.radians(drone.deflector.thrust_yaw_local))
        roll_torque = lever_arm * thrust_force * math.sin(math.radians(drone.deflector.thrust_roll_local))

        return pitch_torque, yaw_torque, roll_torque

    @staticmethod
    def calculate_inertia(drone: DroneSim) -> tuple[float, float]:
        radius = drone.diameter / 2
        pitch_and_yaw_inertia = (drone.weight * radius**2) / 4 + (drone.weight * drone.height**2) / 12
        roll_inertia = (drone.weight * radius**2) / 2
        return pitch_and_yaw_inertia, roll_inertia

    @staticmethod
    def calculate_angular_acceleration(drone: DroneSim):
        """
        Computes angular acceleration based on applied torques and drone inertia.

        Parameters:
            drone: The drone object.
        """

        pitch_torque, yaw_torque, roll_torque = DronePhysicsModel.get_torque(drone)
        pitch_and_yaw_inertia, roll_inertia = DronePhysicsModel.calculate_inertia(drone)
        drone.pitch_acceleration = pitch_torque / pitch_and_yaw_inertia
        drone.yaw_acceleration = yaw_torque / pitch_and_yaw_inertia
        drone.roll_acceleration = roll_torque / roll_inertia

    @staticmethod
    def calculate_linear_acceleration(thrust_x, thrust_y, thrust_z, drone: DroneSim):
        """
        Computes the linear acceleration including thrust, drag, and gravity in the Aerospace (NED) coordinate system.

        Parameters:
            thrust_x (float): Thrust force components (N) in NED.
            thrust_y (float): Thrust force components (N) in NED.
            thrust_z (float): Thrust force components (N) in NED.
            drone: The drone object.

        Returns:
            None
        """

        drag_x, drag_y, drag_z = DronePhysicsModel.calculate_drag(drone)

        acceleration_x = (thrust_x + drag_x) / drone.weight
        acceleration_y = (thrust_y + drag_y) / drone.weight
        acceleration_z = (thrust_z + drag_z) / drone.weight

        acceleration_z -= DronePhysicsModel.GRAVITY

        drone.drone_state.acceleration[:] = [acceleration_x, acceleration_y, acceleration_z]

    @staticmethod
    def calculate_drag(drone: DroneSim) -> tuple[float, float, float]:
        """
        Computes the aerodynamic drag force on the drone.

        Parameters:
            drone: (DroneSim): The drone object.

        Returns:
            drag_x, drag_y, drag_z: Drag forces in each direction (N).
        """
        cd = 0.25
        a = math.pi * (drone.diameter / 2) ** 2

        v_x, v_y, v_z = drone.drone_state.velocity
        v_total = np.linalg.norm(drone.drone_state.velocity)

        rho = DronePhysicsModel.get_air_density(drone.drone_state.position[2])

        drag_force = 0.5 * cd * rho * v_total**2 * a

        if v_total > 1e-6:
            drag_x = -drag_force * (v_x / v_total)
            drag_y = -drag_force * (v_y / v_total)
            drag_z = -drag_force * (v_z / v_total)
        else:
            drag_x, drag_y, drag_z = 0, 0, 0
        return drag_x, drag_y, drag_z