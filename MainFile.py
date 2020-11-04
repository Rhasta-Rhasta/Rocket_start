from MainTypes import *
import numpy as np
import quaternion as qn
from scipy.integrate import solve_ivp


def with_event_attributes(func=None, *, terminal=True, direction=+1):
    if func is None:
        return lambda func: with_event_attributes(func, terminal=terminal, direction=direction)

    assert isinstance(terminal, bool), f"argument 'terminal'={terminal} must have a bool type"
    assert direction in (-1, 0, +1), f"argument 'direction'={direction} must be in (-1, 0, +1)"

    func.terminal = terminal
    func.direction = direction
    return func


@with_event_attributes
def engine_launch(t, y):
    return t - 0.0


@with_event_attributes
def movement_start(t, y):
    rocket.update_state_vector(t, y)
    F = rocket.active_forces('SSK')
    return F[0]


@with_event_attributes
def lift_off(t, y):
    rocket.update_state_vector(t, y)
    return rocket.flight_altitude() - SUPPORT.domain[1]


@with_event_attributes
def stop_integration(t, y):
    rocket.update_state_vector(t, y)
    return rocket.flight_altitude() - 800


def initial_state_vector():
    y0 = np.zeros(21)
    y0[6:9] = np.array([0, -0.21 - 2.325, 0])
    q = qn.from_rotation_vector([
        [0, 0, np.deg2rad(90)],
        # [np.deg2rad(180), 0, 0],
    ])
    y0[9:13] = qn.as_float_array(np.prod(q))
    return y0


def get_events_id(t_events):
    m = np.max([item for item in t_events if item.size > 0])
    res = [i for i, item in enumerate(t_events) if m in item]
    return res


def convert_ode_result(ode_result):
    t = ode_result[0].t
    y = ode_result[0].y
    for i in range(1, len(ode_result)):
        t = np.append(t, ode_result[i].t)
        y = np.hstack((y, ode_result[i].y))
    t, i = np.unique(t, return_index=True)
    y = y[:, i]
    return t, y


def initialization():
    global rocket, time, sv
    rocket = Rocket(initial_state_vector())
    rocket.set_wind_speed(WIND_SPEED)


@timeit
def integration():
    global rocket, time, sv
    events = [
        engine_launch,
        movement_start,
        lift_off,
        stop_integration,
    ]
    step = 1e-1
    res = []
    while stop_integration in events:
        t_eval = 0 if not res else vs.t[-1] + step

        vs = solve_ivp(
            fun=rocket.dydt,
            t_span=(rocket.time, rocket.time + 100),
            y0=rocket.vs,
            method='RK45',
            t_eval=np.arange(t_eval, t_eval + 100, step),
            events=events,
        )

        res.append(vs)

        if vs.status == 1:
            ids = get_events_id(vs.t_events)
            event_time = vs.t_events[ids[0]][-1]
            rocket.update_state_vector(event_time, vs.y_events[ids[0]][-1])
            for i in reversed(ids):
                if 'engine' in events[i].__name__:
                    rocket.set_event(launch_engine=event_time)

                if 'start' in events[i].__name__:
                    rocket.set_event(start_movement=event_time)

                if 'lift' in events[i].__name__:
                    rocket.set_event(lift_off=event_time)

                print(f"{events[i].__name__}: {event_time:0.3f} sec")
                del events[i]

    time, sv = convert_ode_result(res[2:])


def report():
    global rocket, time, sv
    q = qn.from_rotation_vector([
        [0, 0, np.deg2rad(180)],
        [np.deg2rad(90), 0, 0],
    ])
    q = np.prod(q)
    with open('trj.txt', 'w') as file:
        for t, y in zip(time, sv.T):
            rocket.update_state_vector(t, y)
            bot = rocket.polus('ISK') + rocket.bottom_center('ISK')
            bot = qn.rotate_vectors(q, bot)
            quat = q * rocket.Λ
            print(*bot, *qn.as_float_array(quat), *rocket.δ, file=file)


if __name__ == '__main__':
    initialization()
    integration()
    report()
