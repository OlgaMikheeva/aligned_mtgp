import tensorflow as tf
import numpy as np


class EulerODE:
    def __init__(self, f, total_time, nsteps):
        self.ts = np.linspace(0, total_time, nsteps)
        self.f = f

    def forward(self, y0, save_intermediate=False):
        time_grid = tf.constant(self.ts, dtype=tf.float64, name='t')
        time_delta_grid = time_grid[1:] - time_grid[:-1]
        time_grid = time_grid[1:]
        time_combined = \
            tf.concat([time_grid[:, None], time_delta_grid[:, None]], axis=1)
        scan_func = self._make_scan_func(self.f)

        if save_intermediate:
            y_grid = tf.scan(scan_func, time_combined, y0)
            y_s = tf.concat([[y0], y_grid], axis=0)
            y_t = y_s[-1]
            return y_t, y_s
        else:
            y_t = tf.foldl(scan_func, time_combined, y0)
            return y_t, None

    def _step_func(self, evol_func, t_and_dt, y):
        t = t_and_dt[0]
        dt = t_and_dt[1]
        f_sample = evol_func(t, y)
        dt_cast = tf.cast(dt, y.dtype)
        return f_sample * dt_cast

    def _make_scan_func(self, evol_func):
        def scan_func(y, t_and_dt):
            dy = self._step_func(evol_func, t_and_dt, y)
            dy = tf.cast(dy, dtype=y.dtype)
            return y + dy
        return scan_func