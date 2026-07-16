from __future__ import annotations

import math

import newton
import newton.viewer
import numpy as np
import warp as wp


BALL_RADIUS = 0.25
BALL_START_Z = 2.0

CUBE_POSITION = np.array([1.1, 0.0, 0.3], dtype=np.float32)
CUBE_HALF_EXTENT = 0.3
CUBE_ROTATION_STEP = math.radians(90.0)
CUBE_ROTATION_DURATION = 1.25
CUBE_ROTATION_SPEED = CUBE_ROTATION_STEP / CUBE_ROTATION_DURATION

CONTACT_EPS = 0.02
IMPACT_VELOCITY_EPS = -0.05


def z_rotation_quat(angle: float) -> tuple[float, float, float, float]:
    half_angle = 0.5 * angle
    return (0.0, 0.0, math.sin(half_angle), math.cos(half_angle))


def set_cube_pose(state: newton.State, cube_body: int, angle: float, angular_velocity: float) -> None:
    body_q = state.body_q.numpy()
    body_q[cube_body, 0:3] = CUBE_POSITION
    body_q[cube_body, 3:7] = z_rotation_quat(angle)
    state.body_q.assign(body_q)

    body_qd = state.body_qd.numpy()
    body_qd[cube_body, :] = 0.0
    body_qd[cube_body, 5] = angular_velocity
    state.body_qd.assign(body_qd)


def advance_towards(current: float, target: float, speed: float, dt: float) -> tuple[float, float]:
    remaining = target - current
    if abs(remaining) < 1.0e-6:
        return target, 0.0

    step = math.copysign(min(abs(remaining), speed * dt), remaining)
    return current + step, step / dt


def main() -> None:
    builder = newton.ModelBuilder(
        up_axis=newton.Axis.Z,
        gravity=-9.81,
    )

    # Free-falling ball.
    ball_body = builder.add_body(
        xform=wp.transform(
            (0.0, 0.0, BALL_START_Z),
            wp.quat_identity(),
        ),
        mass=1.0,
        label="ball",
    )
    builder.add_shape_sphere(
        body=ball_body,
        radius=BALL_RADIUS,
        color=(0.9, 0.35, 0.25),
    )

    # Side cube. It is kinematic and rotated by the impact event logic below.
    cube_body = builder.add_body(
        xform=wp.transform(
            tuple(CUBE_POSITION),
            wp.quat_identity(),
        ),
        mass=1.0,
        is_kinematic=True,
        label="impact_triggered_cube",
    )
    builder.add_shape_box(
        body=cube_body,
        hx=CUBE_HALF_EXTENT,
        hy=CUBE_HALF_EXTENT,
        hz=CUBE_HALF_EXTENT,
        color=(0.2, 0.55, 0.95),
    )

    builder.add_ground_plane()

    model = builder.finalize()

    solver = newton.solvers.SolverXPBD(
        model,
        iterations=10,
    )

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    newton.eval_fk(
        model,
        model.joint_q,
        model.joint_qd,
        state_0,
    )

    cube_angle = 0.0
    cube_target_angle = 0.0
    set_cube_pose(state_0, cube_body, cube_angle, 0.0)

    viewer = newton.viewer.ViewerGL()
    viewer.set_model(model)
    viewer.set_camera(wp.vec3(3.0, -4.0, 2.2), -25.0, 35.0)

    frame_dt = 1.0 / 60.0
    sim_substeps = 4
    sim_dt = frame_dt / sim_substeps

    sim_time = 0.0
    was_grounded = False

    while viewer.is_running():
        if viewer.should_step():
            for _ in range(sim_substeps):
                previous_ball = state_0.body_q.numpy()[ball_body]
                previous_ball_qd = state_0.body_qd.numpy()[ball_body]
                previous_ball_z = float(previous_ball[2])
                previous_ball_vz = float(previous_ball_qd[2])

                state_0.clear_forces()
                viewer.apply_forces(state_0)
                model.collide(state_0, contacts)

                solver.step(
                    state_0,
                    state_1,
                    control,
                    contacts,
                    sim_dt,
                )

                state_0, state_1 = state_1, state_0

                current_ball_z = float(state_0.body_q.numpy()[ball_body, 2])
                ground_contact_z = BALL_RADIUS + CONTACT_EPS
                is_grounded = current_ball_z <= ground_contact_z
                is_new_impact = (
                    is_grounded
                    and not was_grounded
                    and previous_ball_z > ground_contact_z
                    and previous_ball_vz < IMPACT_VELOCITY_EPS
                )

                if is_new_impact:
                    cube_target_angle += CUBE_ROTATION_STEP

                was_grounded = is_grounded

                cube_angle, cube_angular_velocity = advance_towards(
                    cube_angle,
                    cube_target_angle,
                    CUBE_ROTATION_SPEED,
                    sim_dt,
                )
                set_cube_pose(state_0, cube_body, cube_angle, cube_angular_velocity)

            sim_time += frame_dt

        viewer.begin_frame(sim_time)
        viewer.log_state(state_0)
        viewer.log_contacts(contacts, state_0)
        viewer.end_frame()

    viewer.close()


if __name__ == "__main__":
    main()
