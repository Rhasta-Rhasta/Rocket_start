import polyinterpolation as poly
import numpy as np
import quaternion as qn
from numpy import linalg as la
import inspect
import functools
import time

COORDINATE_SYSTEMS = ('ISK', 'SSK')
CACHE_SIZE = 16
GRAVITY_ACCELERATION = np.array([0, -9.81, 0])
SUPPORT_POINT = np.array([2.325, 0, 0])
TRUST_POINTS = np.array([
    [1.427, +0.8, +0.8],
    [1.427, -0.8, +0.8],
    [1.427, -0.8, -0.8],
    [1.427, +0.8, -0.8],
])

# МЦИХ
MASS = poly.Linear([0, 10, 20, 30], [525.542, 501.539, 477.535, 453.532]) * 1e3
JYZ = poly.Linear([0, 10, 20, 30], [866, 844, 822, 803]) * 1e5
JX = poly.Linear([0, 30], [115750, 115750])
XC = poly.Linear([0, 10, 20, 30], [23.14, 22.95, 22.76, 22.60])

# АДХ
S_MID = 13.2
LENGTH = 57.758
SPEED_OF_SOUND = 343
AIR_DENSITY = 1.2
CY = poly.Linear(np.deg2rad([0, 6, 20, 40, 60, 90, 120, 140, 160, 174, 180]),
                 [0, 0.384, 4, 9, 12, 13.5, 12, 9, 4, 0.5, 0])
CD = poly.Linear(np.deg2rad([0, 6, 20, 40, 60, 90, 120, 140, 160, 174, 180]),
                 [0.81, 0.8, 0.7, 0.55, 0.51, 0.489, 0.440, 0.350, 0.230, 0.20, 0.195])
FT = poly.Linear([0, 0.1, 0.2, 0.3], [4.08, 7.02, 13.87, 24.24]) * 1e3

# ДУ
TRUST = poly.Linear([0, 2.35, 2.5, 3.1, 3.5, 3.8, 4.1, 4.35, 5.35, 14.35, 24.35, 34.35],
                    [0, 0, 94.3, 286.9, 486, 641.1, 717.7, 740.951, 740.999, 742.566, 747.598, 755.832],
                    ) * (1000 * 9.81 / 4)

# усилие подпора
SUPPORT = poly.Linear(np.array([0, 85]) * 1e-3, [660, 660]) * 1e3

# параметры СУ
A0 = poly.Linear([0, 20, 30], [0.4, 0.5, 0.8])
A1 = poly.Linear([0, 20, 30], [1, 1, 0.8])
B0 = 0.06
B1 = 0.8
TAU_1 = 0.023
TAU_2 = 0.21

# траектория
FLIGHT_AZIMUTH = 180  # degrees or None
if FLIGHT_AZIMUTH is not None:
    ROTATE_DURATION = 16  # sec
    ROTATE_START = 8  # sec
    THETA = poly.stack(
        poly.Linear(
            [00.00, 07.00],
            [90.00, 90.00],
        ),
        poly.Spline(
            [07.00, 08.00, 15.00, 16.00, 25.00, 40.00],
            [90.00, 89.96, 87.16, 86.74, 82.07, 72.00],
        ))
    OMEGA_THETA = THETA.diff()

    AZIMUTH = FLIGHT_AZIMUTH
    while abs(AZIMUTH) > 180:
        AZIMUTH -= np.sign(AZIMUTH) * 360
    max_omega = +AZIMUTH / ROTATE_DURATION
    start = ROTATE_START
    duration = ROTATE_DURATION
    OMEGA_PHI = poly.Linear([0, start, start + 2, start + duration, start + duration + 2, 40],
                            [0, 0, max_omega, max_omega, 0, 0])
    PHI = OMEGA_PHI.integr()

# wind speed
WIND_SPEED = np.array([0, 0, 0])  # meters per sec


def timeit(method):
    @functools.wraps(method)
    def timed(*args, **kw):
        ts = time.perf_counter()
        result = method(*args, **kw)
        te = time.perf_counter()
        ms = (te - ts) * 1000
        all_args = ', '.join(tuple(f'{a!r}' for a in args)
                             + tuple(f'{k}={v!r}' for k, v in kw.items()))
        print(f'{method.__name__}({all_args}): {ms:2.2f} ms')
        return result

    return timed


def program_trajectory(t):
    if FLIGHT_AZIMUTH is None:
        q = qn.from_rotation_vector([
            [0, 0, np.deg2rad(90)],
            [np.deg2rad(180), 0, 0],
        ])
        q = np.prod(q)
        omega = np.zeros(3)
    else:
        q = qn.from_rotation_vector([
            [0, 0, np.deg2rad(90)],
            # [np.deg2rad(180), 0, 0],
            [np.deg2rad(-AZIMUTH), 0, 0],
            # [0, 0, np.deg2rad(THETA(t) - 90)],
            [0, 0, np.deg2rad(90 - THETA(t))],
            [np.deg2rad(AZIMUTH + PHI(t)), 0, 0],
        ])
        omega = -(qn.rotate_vectors(q[-1].conj(), [0, 0, np.deg2rad(OMEGA_THETA(t))])
                  + np.array([np.deg2rad(OMEGA_PHI(t)), 0, 0]))
        q = np.prod(q)
    return q, omega


def coord_sys_checker(func):
    sys_checker_enabled = False

    @functools.wraps(func)
    def inner(*args, **kwargs):
        if not inspect.signature(func).parameters.get('coord_sys'):
            raise NotImplementedError(f"There is no argument 'coord_sys' in function '{func.__name__}'")
        sys_id = inspect.getfullargspec(func).args.index('coord_sys')
        sys = kwargs.get('coord_sys') or args[sys_id]
        if sys not in COORDINATE_SYSTEMS:
            raise ValueError(f"'coord_sys'={sys} must be in {COORDINATE_SYSTEMS} calling {func.__name__}")
        return func(*args, **kwargs)

    return inner if sys_checker_enabled else func


def as_matrix(v):
    assert np.shape(v) == (3,), f"Input array shape must be (3,), not {np.shape(v)}')"
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


class Rocket:
    def __init__(self, y):
        self.time = 0
        self.vs = np.zeros(21)

        self.__wind = np.zeros(3)

        # события
        self.event = dict(
            launch_engine=None,
            start_movement=None,
            lift_off=None,
        )

        self.update_state_vector(0, y)
        self.init_bottom_center_pos = self.polus('ISK') + self.bottom_center('ISK')

    @property
    def engine_time(self):
        t = self.event.get('launch_engine') or 0
        return max(0, self.time - t)

    @property
    def movement_time(self):
        t = self.event.get('start_movement') or 0
        return max(0, self.time - t)

    def set_event(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.event:
                self.event[key] = value
            else:
                raise ValueError(f"Wrong argument '{key}', {tuple(self.event.keys())} is possible")

    @property
    def Λ(self):
        return qn.from_float_array(self.vs[9:13])

    @property
    def Λp(self):
        res = []
        for i in range(4):
            q = qn.from_rotation_vector([
                [np.deg2rad(45 + 90 * i), 0, 0],
                [0, 0, np.deg2rad(-4.5)],
                [0, self.δ[i], 0],
            ])
            res.append(np.prod(q))
        return res

    @coord_sys_checker
    def translational_velocity(self, coord_sys):
        res = np.array(self.vs[0:3])
        if coord_sys == 'SSK':
            res = qn.rotate_vectors(self.Λ.conj(), res)
        return res

    @coord_sys_checker
    def angular_velocity(self, coord_sys):
        res = np.array(self.vs[3:6])
        if coord_sys == 'SSK':
            res = qn.rotate_vectors(self.Λ.conj(), res)
        return res

    @coord_sys_checker
    def polus(self, coord_sys):
        res = np.array(self.vs[6:9])
        if coord_sys == 'SSK':
            res = qn.rotate_vectors(self.Λ.conj(), res)
        return res

    @property
    def dδ(self):
        return self.vs[13:17]

    @property
    def δ(self):
        return self.vs[17:21]

    def update_state_vector(self, t, y):
        assert np.shape(y) == (21,)
        # коррекция текущих угловых скоростей и углов поворота камер ДУ
        y[13:17] = np.sign(y[13:17]) * np.min((np.abs(y[13:17]), [np.deg2rad(12)] * 4), axis=0)
        y[17:21] = np.sign(y[17:21]) * np.min((np.abs(y[17:21]), [np.deg2rad(6)] * 4), axis=0)

        # коррекция нормы кватерниона
        y[9:13] = qn.as_float_array(qn.quaternion(*y[9:13]).normalized())

        self.time = t
        self.vs = y.copy()

    @coord_sys_checker
    def inertia_tensor(self, coord_sys):
        m = MASS(self.movement_time)
        rc = self.center_of_mass('SSK')
        jx = JX(self.movement_time)
        jy = jz = JYZ(self.movement_time)
        J = np.diagflat([jx, jy, jz])
        J = J + m * (np.eye(3) * np.inner(rc, rc) - np.outer(rc, rc))
        if coord_sys == 'ISK':
            J = qn.rotate_vectors(self.Λ, J.T)
            J = qn.rotate_vectors(self.Λ, J.T)
            rc = qn.rotate_vectors(self.Λ, rc)
        s = as_matrix(m * rc)
        m = np.diagflat([m] * 3)
        return np.block([[m, -s],
                         [s, J]])

    @coord_sys_checker
    def center_of_mass(self, coord_sys):
        res = np.array([XC(self.movement_time), 0, 0])
        if coord_sys == 'ISK':
            res = qn.rotate_vectors(self.Λ, res)
        return res

    @coord_sys_checker
    def support_point(self, coord_sys):
        rd = self.bottom_center('SSK')
        res = rd + SUPPORT_POINT
        if coord_sys == 'ISK':
            res = qn.rotate_vectors(self.Λ, res)
        return res

    @coord_sys_checker
    def bottom_center(self, coord_sys):
        rd = np.zeros(3)
        if coord_sys == 'ISK':
            rd = qn.rotate_vectors(self.Λ, rd)
        return rd

    @coord_sys_checker
    def trust_points(self, coord_sys):
        rd = self.bottom_center('SSK')
        res = rd + TRUST_POINTS
        if coord_sys == 'ISK':
            res = qn.rotate_vectors(self.Λ, res)
        return res

    @coord_sys_checker
    def gravity_acceleration(self, coord_sys):
        res = GRAVITY_ACCELERATION
        if coord_sys == 'SSK':
            res = qn.rotate_vectors(self.Λ.conj(), res)
        return res

    @coord_sys_checker
    def trust(self, coord_sys):
        if self.event['launch_engine'] is None:
            return np.zeros(6)
        else:
            p = [TRUST(self.engine_time), 0, 0]
            rp = self.trust_points('SSK')
            force = []
            for q in self.Λp:
                force.append(qn.rotate_vectors(q, p))
            moment = np.cross(rp, force)
            force = np.sum(np.array(force), axis=0)
            moment = np.sum(np.array(moment), axis=0)
            if coord_sys == 'ISK':
                force, moment = qn.rotate_vectors(self.Λ, (force, moment))
            return np.append(force, moment)

    def flight_altitude(self):
        rd = self.polus('ISK') + self.bottom_center('ISK')
        delta = rd - self.init_bottom_center_pos
        return delta[1]

    @coord_sys_checker
    def support_force(self, coord_sys):
        if self.event['lift_off'] is not None:
            return np.zeros(6)
        else:
            alt = np.clip(self.flight_altitude(), *SUPPORT.domain)
            force = np.array([0, SUPPORT(alt), 0])
            if coord_sys == 'SSK':
                force = qn.rotate_vectors(self.Λ.conj(), force)
            rs = self.support_point(coord_sys)
            moment = np.cross(rs, force)
            return np.append(force, moment)

    @coord_sys_checker
    def wind_speed(self, coord_sys):
        res = self.__wind
        if coord_sys == 'SSK':
            res = qn.rotate_vectors(self.Λ.conj(), res)
        return res

    def set_wind_speed(self, speed):
        self.__wind = np.array(speed)

    @coord_sys_checker
    def relative_velocity(self, coord_sys):
        v = self.translational_velocity(coord_sys)
        w = self.wind_speed(coord_sys)
        return v - w

    @coord_sys_checker
    def relative_velocity_normalize(self, coord_sys):
        res = self.relative_velocity(coord_sys)
        module = la.norm(res)
        return res / module if module else res

    def attack_angle(self):
        Ve = self.relative_velocity_normalize('SSK')
        angle = np.arccos(np.inner(Ve, [1, 0, 0]))
        return angle

    def mah(self):
        Ve = self.relative_velocity('ISK')
        return la.norm(Ve) / SPEED_OF_SOUND

    def velocity_head(self):
        Ve = self.relative_velocity('ISK')
        return AIR_DENSITY * la.norm(Ve) ** 2 / 2

    @coord_sys_checker
    def aerodynamic_focus(self, coord_sys):
        rd = self.bottom_center('SSK')
        alpha = self.attack_angle()
        focus = np.array([CD(alpha) * LENGTH, 0, 0])
        res = rd + focus
        if coord_sys == 'ISK':
            res = qn.rotate_vectors(self.Λ, res)
        return res

    @coord_sys_checker
    def aerodynamic_force(self, coord_sys):
        rf = self.aerodynamic_focus('SSK')
        Ve = self.relative_velocity_normalize('SSK')
        mah = self.mah()
        Xa = FT(mah)
        ex = np.array([1, 0, 0])
        alpha = self.attack_angle()
        q = self.velocity_head()
        cy = CY(alpha)
        Ya = cy * q * S_MID
        eyz = np.cross(ex, np.cross(ex, Ve))
        force = -ex * Xa + eyz * Ya
        moment = np.cross(rf, force)
        if coord_sys == 'ISK':
            force, moment = qn.rotate_vectors(self.Λ, (force, moment))
        return np.append(force, moment)

    @coord_sys_checker
    def gravity_force(self, coord_sys):
        m = MASS(self.movement_time)
        g = self.gravity_acceleration(coord_sys)
        rc = self.center_of_mass(coord_sys)
        force = m * g
        moment = np.cross(rc, force)
        return np.append(force, moment)

    @coord_sys_checker
    def active_forces(self, coord_sys):
        P = self.trust(coord_sys)
        Fa = self.aerodynamic_force(coord_sys)
        mg = self.gravity_force(coord_sys)
        Fs = self.support_force(coord_sys)
        return P + Fa + mg + Fs

    @coord_sys_checker
    def inertia_force(self, coord_sys):
        ω = self.angular_velocity('SSK')
        tensor = self.inertia_tensor('SSK')
        J = tensor[3:6, 3:6]
        s = np.array([tensor[1, 5], -tensor[0, 5], tensor[0, 4]])
        force = -np.cross(ω, np.cross(ω, s))
        moment = -np.cross(ω, J @ ω)
        if coord_sys == 'ISK':
            force, moment = qn.rotate_vectors(self.Λ, (force, moment))
        return np.append(force, moment)

    def acceleration(self):
        T = self.inertia_tensor('ISK')
        F = (self.active_forces('ISK') +
             self.inertia_force('ISK'))

        if self.event['start_movement'] is None:
            T = np.block([[T, np.eye(6)],
                          [np.eye(6), np.zeros((6, 6))]])
            F = np.append(F, np.zeros(6))

        res = la.solve(T, F)
        return res[0:6]

    def dΛ(self):
        omega = self.angular_velocity('ISK')
        omega = qn.quaternion(*omega)
        res = omega * self.Λ / 2
        return qn.as_float_array(res)

    def rudders(self):
        if self.event['start_movement'] is not None:
            orientation = self.Λ
            angular_velocity = self.angular_velocity('SSK')
            a0 = np.array([B0, A0(self.movement_time), A0(self.movement_time)])
            a1 = np.array([B1, A1(self.movement_time), A1(self.movement_time)])
            program_orientation, program_angular_velocity = program_trajectory(self.movement_time)
            delta_angle = qn.as_rotation_vector(program_orientation.conj() * orientation)
            delta_angular_velocity = angular_velocity - program_angular_velocity
            signal = a0 * (delta_angle + a1 * delta_angular_velocity)
            signal = np.array([[1, +np.sqrt(2), +np.sqrt(2)],
                               [1, -np.sqrt(2), +np.sqrt(2)],
                               [1, -np.sqrt(2), -np.sqrt(2)],
                               [1, +np.sqrt(2), -np.sqrt(2)]]) @ signal
            res = np.concatenate([self.dδ, (signal - TAU_2 * self.dδ - self.δ) / TAU_1])
        else:
            res = np.zeros(8)
        return res

    def dydt(self, t, y):
        self.update_state_vector(t, y)
        eps = self.acceleration()
        dr = self.translational_velocity('ISK')
        dΛ = self.dΛ()
        δ = self.rudders()
        res = np.concatenate((eps, dr, dΛ, δ))
        return res


if __name__ == "__main__":
    pass
