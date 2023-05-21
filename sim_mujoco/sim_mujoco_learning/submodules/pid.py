import math

class IIR_low_pass:
    def __init__(self, Fs, Fc):
        self.y_prev = 0
        delta_t = 1/Fs
        if math.isinf(Fc):
            self.alpha = 1
        else:
            self.alpha = 2*math.pi * delta_t * Fc / (2*math.pi * delta_t * Fc + 1)

    def update(self, x):
        y = self.alpha * x + (1-self.alpha)*self.y_prev
        self.y_prev = y
        return y


class pid:

    def __init__(self):
        self.pid_set_gains(0.0, 0.0, 0.0)
        self.integrator = 0.0
        self.previous_error = 0.0
        self.integrator_limit = math.inf
        self.frequency = 1.0
        self.d_lp_fc = math.inf
        self.d_lp = IIR_low_pass(self.frequency, self.d_lp_fc)
        self.output_max = math.inf

    def pid_set_gains(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def pid_get_gains(self):
        return self.kp, self.ki, self.kd

    def pid_set_integral_limit(self, integral_max):
        self.integrator_limit = integral_max

    def pid_output_limit(self, output_max):
        self.output_max = output_max

    def pid_get_integral_limit(self):
        return self.integrator_limit

    def pid_get_integral(self):
        return self.integrator

    def pid_reset(self):
        self.integrator = 0.
        self.previous_error = 0.

    def pid_set_frequency(self, frequency):
        self.frequency = frequency
        self.d_lp = IIR_low_pass(self.frequency, self.d_lp_fc)

    def pid_get_frequency(self):
        return self.frequency

    def set_derivative_term_low_pass_frequency(self, fc):
        self.d_lp_fc = fc
        self.d_lp = IIR_low_pass(self.frequency, self.d_lp_fc)

    def pid_process(self, error):
        if not math.isfinite(error):
            return 0

        self.integrator = self.integrator + error

        if self.ki > 0:
            out_saturation_integrator_limit = self.output_max / self.ki * self.frequency
        else:
            out_saturation_integrator_limit = 0
        integrator_limit = min(out_saturation_integrator_limit, self.integrator_limit)
        if (self.integrator > integrator_limit):
            self.integrator = integrator_limit
        elif self.integrator < - integrator_limit:
                self.integrator = - integrator_limit

        output = - self.kp * error
        output += -self.ki * self.integrator / self.frequency
        output += -self.kd * self.d_lp.update(error - self.previous_error) * self.frequency
        self.previous_error = error

        return output
