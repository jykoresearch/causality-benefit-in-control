
import numpy as np


class ThermalComfortCalculator:
    def __init__(self):
        pass

    def calc_skin_temp(self, M, W):
        """Calculate comfort skin temperature (K)"""
        return 308.7 - 0.0275 * (M - W)

    def calc_sweat(self, M, W):
        """Calculate comfort sweat rate (W/m²)"""
        return 0.42 * (M - W - 58.15)

    def calc_rad(self, fcl, Tcl, MRT, d1=0):
        """Calculate radiation heat loss (W/m²)"""
        return (1 + d1) * 3.96e-8 * fcl * (Tcl ** 4 - MRT ** 4)

    def calc_conv(self, fcl, hc, Tcl, Tair):
        """Calculate convection heat loss (W/m²)"""
        return fcl * hc * (Tcl - Tair)

    def calc_evap(self, M, W, pa):
        """Calculate evaporation heat loss (W/m²)"""
        return 3.05 * (5.73 - 0.007 * (M - W) - pa / 1000)

    def calc_resp(self, M, Tair, pa):
        """Calculate respiration heat loss (W/m²)"""
        return 0.0014 * M * (34 - (Tair - 273)) + 0.0173 * M * (5.87 - pa / 1000)

    def calc_hc(self, Tcl, Tair, Vel, d2=0, d3=0):
        """Calculate convective heat transfer coefficient (W/m²K)"""
        term_1 = (1 + d2) * 2.38 * abs(Tcl - Tair) ** 0.25
        term_2 = (1 + d3) * 12.1 * np.sqrt(Vel)
        return np.maximum(term_1, term_2)

    def calc_fcl(self, Icl):
        """Calculate clothing area factor"""
        if Icl < 0.5:
            return 1.0 + 0.2 * Icl
        else:
            return 1.05 + 0.1 * Icl

    def calc_pa(self, RH, Tair):
        """Calculate water vapor pressure (Pa)"""
        result = RH * 10 * np.exp((16.6536 - 4030.183 / (Tair - 273 + 235)))
        return result

    def calc_cloth_temp(self, Tskin_comfort, Icl, h_radiation, h_convection, d4=0):
        """Calculate clothing temperature (K)"""
        return Tskin_comfort - 0.155 * Icl * (h_radiation + h_convection) + d4

    def calc_heat_loss(self, h_radiation, h_convection, h_evaporation, h_respiration, h_sweat, h_adj_loss):
        """Calculate total heat loss (W/m²)"""
        return h_radiation + h_convection + h_evaporation + h_respiration + h_sweat + h_adj_loss

    def calc_adj_loss(self, M, W, d5=0):
        return d5 * (M - W - 58.15)

    def calc_therm_load(self, M, W, h_loss):
        """Calculate thermal load (W/m²)"""
        return (M - W) - h_loss

    def calc_comfort(self, Tair, MRT, RH, M = 58.15, W = 0, Icl = 0.5, Vel = 0.1, d1=0, d2=0, d3=0, d4=0, d5=0):
        Tskin_comfort = self.calc_skin_temp(M, W)
        h_sweat = self.calc_sweat(M, W)
        fcl = self.calc_fcl(Icl)
        pa = self.calc_pa(RH, Tair)
        Tcl = Tskin_comfort

        for _ in range(10):
            hc = self.calc_hc(Tcl, Tair, Vel, d2, d3)
            h_radiation = self.calc_rad(fcl, Tcl, MRT, d1)
            h_convection = self.calc_conv(fcl, hc, Tcl, Tair)
            Tcl_new = self.calc_cloth_temp(Tskin_comfort, Icl, h_radiation, h_convection, d4)

            if abs(Tcl - Tcl_new) < 0.1:
                break

            Tcl = Tcl_new

        h_evaporation = self.calc_evap(M, W, pa)
        h_respiration = self.calc_resp(M, Tair, pa)

        h_adj_loss = self.calc_adj_loss(M, W, d5)
        h_loss = self.calc_heat_loss(h_radiation, h_convection, h_evaporation, h_respiration, h_sweat, h_adj_loss)

        thermal_load = self.calc_therm_load(M, W, h_loss)

        return thermal_load

class OB_model:
    def __init__(self, b1 = 0.15, b0 = 2.3, s_a = 0.25, s_0 = -22, sen_dt = 1.5):            
        ''' 
        b0: One's thermal condition under full satisfaction
        '''
        self.T = 26
        self.MRT = 26
        self.RH = 50
        self.acc_disc = 0
        self.p_override = 0
        self.override = 0 
        self.dist_mu = 0
        self.dt = 0
        self.dt_obs = 0
        self.overriden_period = 0
        self.hold_timer = 0
        self.hold_temp = 27
        self.term_dist_mu = 0
        self.term_dist_pattern = 0
        self.term_beta0 = 0 
        self.E = 0
        self.s_0 = s_0
        self.s_a = s_a
        self.b1 = b1
        self.b0 = b0
        self.sen_dt = sen_dt
        self.timestep_sim = 15 
        self.term_acc_disc = 0

    def override_behavior(self, T, MRT, RH):

        self.E = ThermalComfortCalculator().calc_comfort(
            M=58.15,
            W=0,
            Tair=273 + T,
            MRT=273 + MRT,
            Vel=0.1,
            RH= RH,
            Icl=0.5)

        self.dist_mu = np.random.normal(loc= self.b1 * self.E + self.b0, scale=0.01)

        self.term_dist_mu = np.abs(self.dist_mu)
        self.term_acc_disc = abs(self.acc_disc)

        self.p_override = 1 / (1 + np.exp(-(self.term_dist_mu + self.term_acc_disc + self.s_0)))
        self.override = np.random.binomial(1, self.p_override)

        dt = (np.random.normal(loc=-self.sen_dt * self.dist_mu, scale=0.1))
        self.dt = round(dt / 0.5) * 0.5
        self.dt_obs = self.dt * self.override
        self.acc_disc = (1 - self.override) * (self.s_a * (self.acc_disc +  self.timestep_sim * self.dist_mu)) 

    def time_goes(self):
        self.hold_timer += 1

    def reset_hold_timer(self):
        self.hold_timer = 0

    def update_hold_temp(self, T):
        self.hold_temp = T

if __name__ == "__main__":
    calc = ThermalComfortCalculator()
    load_cool = calc.calc_comfort(Tair=273 + 22, MRT=273 + 22, RH=50)
    load_warm = calc.calc_comfort(Tair=273 + 30, MRT=273 + 30, RH=50)
    assert load_warm > load_cool, (load_cool, load_warm)

    ob = OB_model()
    ob.override_behavior(T=30, MRT=30, RH=50)
    p_warm = ob.p_override
    ob = OB_model()
    ob.override_behavior(T=22, MRT=22, RH=50)
    assert p_warm > ob.p_override, (ob.p_override, p_warm)
    assert 0 <= ob.p_override <= 1

    ob.time_goes()
    assert ob.hold_timer == 1
    ob.reset_hold_timer()
    assert ob.hold_timer == 0
    print("ok")
