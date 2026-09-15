# Portions of this file are derived from airboxlab/rllib-energyplus
# (rleplus/env/energyplus.py), MIT License, Copyright (c) 2022 Antoine Galataud.
# Full license text and the scope of what is derived: see README.md
# Source: https://github.com/airboxlab/rllib-energyplus

"""Gymnasium environment wrapping EnergyPlus for behaviour-aware setpoint control."""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import abc
import os
import threading
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Dict, List, Optional, Tuple, Union

import signal
from pyenergyplus.api import EnergyPlusAPI
from pyenergyplus.datatransfer import DataExchange
import torch

_ROOT = Path(__file__).resolve().parent
IDF_DIR = _ROOT
DATA_DIR = _ROOT

class IDF_handler():
    def __init__(self, dir : str = str(IDF_DIR / "00_building_model.idf")):
        self.raw_file_dir = dir
        assert os.path.isfile(self.raw_file_dir), f"The file {self.raw_file_dir} does not exist."
        file = open(self.raw_file_dir, "r")
        self.raw_file = file.read()


    def edit_and_save(self,
                      start_month: int, start_day: int,
                      end_month: int, end_day: int):
        """Write the run-period IDF for the given start/end dates."""
        self.raw_file_formatted = self.raw_file.format(start_month=start_month, start_day=start_day, end_month=end_month, end_day=end_day)

        dir = os.path.join(str(IDF_DIR), "00_building_model_{}_{}_{}_{}.idf").format(start_month, start_day, end_month, end_day)
        with open(dir, "w") as file:
            file.write(self.raw_file_formatted)

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


class RunnerConfig:
    """Configuration for the runner."""

    epw: Union[Path, str]
    idf: Union[Path, str]
    output: Union[Path, str]
    variables: Dict[str, Tuple[str, str]]
    meters: Dict[str, str]
    actuators: Dict[str, Tuple[str, str, str]]
    csv: bool = False
    verbose: bool = False
    
    def __init__(self, epw, idf, output, variables, meters, actuators, csv, verbose):
        self.epw = epw
        self.idf = idf
        self.output = output
        self.variables = variables
        self.meters = meters
        self.actuators = actuators
        self.csv = csv
        self.verbose = verbose


class EnergyPlusRunner:
    """EnergyPlus simulation runner.

    This class is responsible for running EnergyPlus in a separate thread and to interact
    with it through its API.
    """

    def __init__(self, episode: int, obs_queue: Queue, act_queue: Queue, runner_config: RunnerConfig, 
                rl_option, occ_option, save_csv, weather_file, model_name, obj_coeffs,
                reward_form, rl_occ_obs, baseline_temp,
                dd_ob_model, scaler, ob_causal, 
                b1 = 0.15, b0 = 2.3, s_a = 0.3, s_0 = -18, sen_dt = 2, occ_case = 0,
                hold_duration = None,
                ) -> None:

        self.override_duration = 48
        self.b1 = b1
        self.b0 = b0
        self.s_0 = s_0
        self.sen_dt = sen_dt
        self.s_a = s_a
        self.hold_status = 0
        self.on_off_flag = 0

        if occ_case == 1:
            b1 = 0.15; b0 = 2.8; s_a = 0.3; s_0 = -18; sen_dt = 2;
            self.b1= b1; self.b0 = b0; self.s_a = s_a; self.s_0 = s_0; self.sen_dt = sen_dt

        if occ_case == 2:
            b1 = 0.15; b0 = 2.3; s_a = 0.4; s_0 = -18; sen_dt = 2;
            self.b1= b1; self.b0 = b0; self.s_a = s_a; self.s_0 = s_0; self.sen_dt = sen_dt

        if occ_case == 3:
            b1 = 0.15; b0 = 2.3; s_a = 0.3; s_0 = -14; sen_dt = 2;
            self.b1= b1; self.b0 = b0; self.s_a = s_a; self.s_0 = s_0; self.sen_dt = sen_dt

        if occ_case == 4:
            b1 = 0.15; b0 = 2.3; s_a = 0.3; s_0 = -18; sen_dt = 1.5;
            self.b1= b1; self.b0 = b0; self.s_a = s_a; self.s_0 = s_0; self.sen_dt = sen_dt

        self.occupant = OB_model(b1 = self.b1, b0 = self.b0, s_a = self.s_a, s_0= s_0, sen_dt = self.sen_dt)

        self.hold_duration = hold_duration
        if self.hold_duration is not None:
            self.occupant.hold_timer = self.hold_duration
        self.episode = episode
        self.runner_config = runner_config
        self.verbose = self.runner_config.verbose

        self.obs_queue = obs_queue
        self.act_queue = act_queue
        self.act_queue_mutex = threading.Lock()

        self.energyplus_api = EnergyPlusAPI()
        self.x: DataExchange = self.energyplus_api.exchange
        self.energyplus_exec_thread: Optional[threading.Thread] = None
        self.energyplus_state: Any = None
        self.sim_results: Dict[str, Any] = {}
        self.initialized = False
        self.progress_value: int = 0
        self.simulation_complete = False

        self.variables = runner_config.variables
        self.var_handles: Dict[str, int] = {}

        self.meters = runner_config.meters
        self.meter_handles: Dict[str, int] = {}

        self.actuators = runner_config.actuators
        self.actuator_handles: Dict[str, int] = {}
        self.last_action = 0.0
        self.cooling_setpoint_prev = 25
        self.heating_setpoint = -20
        self.cooling_setpoint = 26
        self.save_csv = save_csv
        
        self.rl_option = rl_option
        self.occ_option = occ_option
        self.model_name = model_name

        self.weather = weather_file
        self.weather["time_key"] = self.weather["Time"].round(2)
        self.weather_index_map = {
            (int(row["day_of_year"]), float(row["time_key"])): idx
            for idx, row in self.weather.iterrows()
        }

        self.Ti_prev = 0.5
        self.init = 0
        self.obj_coeffs = obj_coeffs

        self.reward_form = reward_form
        self.rl_occ_obs = rl_occ_obs
        self.baseline_temp = baseline_temp
        self.dd_ob_model = dd_ob_model
        self.scaler = scaler
        
        self.ob_causal = ob_causal
        
        self.override_prev = 0 
        self.hold_status_prev = 0
        self.hold_timer_prev = 0

        self.thermostat_program = 0

        if self.ob_causal == 0: # Non-causal model
            self.x_input = np.zeros((8, 20))

        elif self.ob_causal == 4: # Causal model
            self.x_input = np.zeros((8, 4))

        self.df = pd.DataFrame({})

        self.dd_ob_model.eval() if self.dd_ob_model is not None else None

    def occupant_simulation(self, Ti, mrt, rh, time):
        
        self.hold_status = 0

        if self.thermostat_program != thermostat_activity(time):
            self.hold_status_prev = 0 
            self.thermostat_program = thermostat_activity(time)

        if self.hold_duration is None:
            hold_active = (self.hold_status_prev == 1)
        else:
            hold_active = (self.occupant.hold_timer < self.hold_duration)

        if self.dd_ob_model is None:
            self.occupant.override_behavior(T=Ti, MRT=mrt, RH=rh)
            
            if self.occ_option == 1:
                if self.occupant.override == 1 and self.occupant.dt != 0:
                    self.hold_status = 1 
                    self.occupant.update_hold_temp(self.cooling_setpoint_prev + self.occupant.dt)
                    self.cooling_setpoint = self.occupant.hold_temp
                    self.occupant.reset_hold_timer()

                elif hold_active: 
                    self.cooling_setpoint = self.occupant.hold_temp
                    self.hold_status = 1       

                self.occupant.time_goes()

            else:
                self.occupant.override = 0        
                self.occupant.hold_timer = 0

        else:
            
            torch_x_input = torch.tensor(self.x_input, dtype=torch.float32).unsqueeze(0) 

            if torch_x_input.size(1) == 8:
                with torch.no_grad():
                    self.predicted_label, _ = self.dd_ob_model.predict_proba(torch_x_input)                
                self.predicted_dt = 0.5*float(self.predicted_label[0]-19)

            else:
                self.predicted_dt = 0

            if self.predicted_dt != 0: 
                self.hold_status = 1
                self.occupant.update_hold_temp(self.cooling_setpoint_prev + self.predicted_dt)
                self.cooling_setpoint = self.occupant.hold_temp
                self.occupant.reset_hold_timer()

            elif hold_active: 
                self.cooling_setpoint = self.occupant.hold_temp
                self.hold_status = 1       

            self.occupant.override = 1 if self.predicted_dt != 0 else 0
            self.occupant.dt_obs = self.predicted_dt
            self.occupant.time_goes()

    def start(self) -> None:
        self.energyplus_state = self.energyplus_api.state_manager.new_state()
        runtime = self.energyplus_api.runtime
        runtime.set_console_output_status(self.energyplus_state, False)

        def _report_progress(progress: int) -> None:
            self.progress_value = progress
            if self.verbose:
                print(f"Simulation progress: {self.progress_value}%")

        runtime.callback_progress(self.energyplus_state, _report_progress)

        runtime.callback_begin_zone_timestep_after_init_heat_balance(self.energyplus_state, self._collect_obs)

        def _run_energyplus(rn, cmd_args, state, results):
            if self.verbose:
                print(f"running EnergyPlus with args: {cmd_args}")
        
            results["exit_code"] = rn.run_energyplus(state, cmd_args)
            

            if not self.simulation_complete:
                self.obs_queue.put(None)
                self.act_queue.put(None)
                self.stop()
            
        self.energyplus_exec_thread = threading.Thread(
            target=_run_energyplus,
            args=(self.energyplus_api.runtime, self.make_eplus_args(), self.energyplus_state, self.sim_results),
        )
        self.energyplus_exec_thread.start()

    def stop(self) -> None:
        if not self.simulation_complete:
            self.simulation_complete = True
            self._flush_queues()

            if threading.current_thread() != self.energyplus_exec_thread:
                self.energyplus_exec_thread.join()
            
            self.energyplus_exec_thread = None
            self.energyplus_api.runtime.clear_callbacks()
            self.energyplus_api.state_manager.delete_state(self.energyplus_state)
            

    def failed(self) -> bool:
        return self.sim_results.get("exit_code", -1) > 0

    def make_eplus_args(self) -> List[str]:
        """Make command line arguments to pass to EnergyPlus."""
        eplus_args = ["-r"] if self.runner_config.csv else []
        eplus_args += [
            "-w",
            self.runner_config.epw,
            "-d",
            f"{self.runner_config.output}/episode-{self.episode:08}-{os.getpid():05}",
            self.runner_config.idf
        ]
        return eplus_args

    def init_exchange(self, default_action: float) -> Dict[str, float]:
        self.last_action = default_action
        self.act_queue.put(default_action)
        return self.obs_queue.get()

    def _collect_obs(self, state_argument) -> None:
        """EnergyPlus callback that collects output variables/meters values and enqueue them."""
        if self.simulation_complete or not self._init_callback(state_argument):
            return


        time = self.x.current_time(state_argument); 
        time = np.floor(time * 4) / 4
        time_sin = np.sin(2*np.pi*time/24); time_cos = np.cos(2*np.pi*time/24)
        day_of_year = self.x.day_of_year(state_argument)
        day_of_week = self.x.day_of_week(state_argument)

        day_of_year_sin = np.sin(2*np.pi*day_of_year/365); day_of_year_cos = np.cos(2*np.pi*day_of_year/365)
        day_of_week_sin = np.sin(2*np.pi*day_of_week/7); day_of_week_cos = np.cos(2*np.pi*day_of_week/7)


        lookup_key = (int(day_of_year), round(time, 2))
        row_number = self.weather_index_map.get(lookup_key)

        if row_number is None:
            for delta in [-0.01, 0.01, -0.02, 0.02]:
                alt_key = (int(day_of_year), round(time + delta, 2))
                row_number = self.weather_index_map.get(alt_key)
                if row_number is not None:
                    break
        
        weather_one_hour_later = self.weather.loc[row_number + 12]
        weather_two_hours_later = self.weather.loc[row_number + 24]
        
        
        if self.rl_occ_obs == 1:

            self.next_obs = {
                **{key: self.x.get_variable_value(state_argument, handle) for key, handle in self.var_handles.items()},
                **{key: self.x.get_meter_value(state_argument, handle) * 0.00000028 if key == "elec_hvac" else self.x.get_meter_value(state_argument, handle) for key, handle in self.meter_handles.items()},
                "hold_status" : self.hold_status,
                "occupant_override" : self.occupant.override,
                "hold_timer": self.occupant.hold_timer,
                "To_pred_one" : weather_one_hour_later["To"],
                "RHo_pred_one" : weather_one_hour_later["RHo"],
                "I_dir_pred_one" : weather_one_hour_later["I_dir"],
                "I_dif_pred_one" : weather_one_hour_later["I_dif"],
                "I_gr_pred_one" : weather_one_hour_later["I_gr"],
                "v_wind_pred_one" : weather_one_hour_later["v_wind"],
                "To_pred_two" : weather_two_hours_later["To"],
                "RHo_pred_two" : weather_two_hours_later["RHo"],
                "I_dir_pred_two" : weather_two_hours_later["I_dir"],
                "I_dif_pred_two" : weather_two_hours_later["I_dif"],
                "I_gr_pred_two" : weather_two_hours_later["I_gr"],
                "v_wind_pred_two" : weather_two_hours_later["v_wind"],
                "time_sin": time_sin,
                "time_cos": time_cos,
                "day_of_year_sin": day_of_year_sin,
                "day_of_year_cos": day_of_year_cos,
                "day_of_week_sin": day_of_week_sin,
                "day_of_week_cos": day_of_week_cos
                }
        
        else: 

            self.next_obs = {
            **{key: self.x.get_variable_value(state_argument, handle) for key, handle in self.var_handles.items()},
            **{key: self.x.get_meter_value(state_argument, handle) * 0.00000028 if key == "elec_hvac" else self.x.get_meter_value(state_argument, handle) for key, handle in self.meter_handles.items()},
            "To_pred_one" : weather_one_hour_later["To"],
            "RHo_pred_one" : weather_one_hour_later["RHo"],
            "I_dir_pred_one" : weather_one_hour_later["I_dir"],
            "I_dif_pred_one" : weather_one_hour_later["I_dif"],
            "I_gr_pred_one" : weather_one_hour_later["I_gr"],
            "v_wind_pred_one" : weather_one_hour_later["v_wind"],
            "To_pred_two" : weather_two_hours_later["To"],
            "RHo_pred_two" : weather_two_hours_later["RHo"],
            "I_dir_pred_two" : weather_two_hours_later["I_dir"],
            "I_dif_pred_two" : weather_two_hours_later["I_dif"],
            "I_gr_pred_two" : weather_two_hours_later["I_gr"],
            "v_wind_pred_two" : weather_two_hours_later["v_wind"],
            "time_sin": time_sin,
            "time_cos": time_cos,
            "day_of_year_sin": day_of_year_sin,
            "day_of_year_cos": day_of_year_cos,
            "day_of_week_sin": day_of_week_sin,
            "day_of_week_cos": day_of_week_cos
            }

        Ti = self.next_obs["Ti"]
        mrt = self.next_obs["mrt"] 
        rh = self.next_obs["rh"]
        outdoor_temp = self.next_obs["outdoor_temp"]
        outdoor_rh = self.next_obs["outdoor_rh"]
        direct_solar = self.next_obs["direct_solar"]
        diffuse_solar = self.next_obs["diffuse_solar"]
        ground_solar = self.next_obs["ground_solar"]
        wind_speed = self.next_obs["wind_speed"]
        elec_hvac = self.next_obs["elec_hvac"]


        elec_fans_1 = self.x.get_meter_value(state_argument, self.x.get_meter_handle(state_argument, "Fans:Electricity"))* 0.00000028
        elec_hvac = self.x.get_meter_value(state_argument, self.x.get_meter_handle(state_argument, "Electricity:HVAC"))* 0.00000028
        cooling_capa = self.x.get_meter_value(state_argument, self.x.get_meter_handle(state_argument, "CoolingCoils:EnergyTransfer"))* 0.00000028

                
        try:
            cop = cooling_capa / elec_hvac
        except:
            cop = 0

        ranges = {
                    "Ti": (20, 30),
                    "mrt": (0, 40),
                    "rh": (0, 100),
                    "outdoor_temp": (0, 40),
                    "outdoor_rh": (0, 100),
                    "direct_solar": (0, 1000),
                    "diffuse_solar": (0, 1000),
                    "ground_solar": (0, 1000),
                    "wind_speed": (0, 15),
                    "elec_hvac": (0, 0.7), 
                    "hold_timer" : (0, 2000),
                    "To_pred_one" : (0, 40),
                    "RHo_pred_one" : (0, 100),
                    "I_dir_pred_one" : (0, 1000),
                    "I_dif_pred_one" : (0, 1000),
                    "I_gr_pred_one" : (0, 1000),
                    "v_wind_pred_one" : (0, 15),
                    "To_pred_two" : (0, 40),
                    "RHo_pred_two" : (0, 100),
                    "I_dir_pred_two" : (0, 1000),
                    "I_dif_pred_two" : (0, 1000),
                    "I_gr_pred_two" : (0, 1000),
                    "v_wind_pred_two" : (0, 15)
                }
        
        if self.rl_occ_obs == 0:
            ranges = {
                    "Ti": (20, 30),
                    "mrt": (0, 40),
                    "rh": (0, 100),
                    "outdoor_temp": (0, 40),
                    "outdoor_rh": (0, 100),
                    "direct_solar": (0, 1000),
                    "diffuse_solar": (0, 1000),
                    "ground_solar": (0, 1000),
                    "wind_speed": (0, 15),
                    "elec_hvac": (0, 0.7), 
                    "To_pred_one" : (0, 40),
                    "RHo_pred_one" : (0, 100),
                    "I_dir_pred_one" : (0, 1000),
                    "I_dif_pred_one" : (0, 1000),
                    "I_gr_pred_one" : (0, 1000),
                    "v_wind_pred_one" : (0, 15),
                    "To_pred_two" : (0, 40),
                    "RHo_pred_two" : (0, 100),
                    "I_dir_pred_two" : (0, 1000),
                    "I_dif_pred_two" : (0, 1000),
                    "I_gr_pred_two" : (0, 1000),
                    "v_wind_pred_two" : (0, 15)
                }

        for key, (min_val, max_val) in ranges.items():
            self.next_obs[key] = (self.next_obs[key] - min_val) / (max_val - min_val)
        
        if self.init == 0:
            self.init = 1
            self.next_obs["Ti_change"] = 0
        else: 
            self.next_obs["Ti_change"] = self.next_obs["Ti"] - self.Ti_prev

        self.Ti_prev = self.next_obs["Ti"]

        self.obs_queue.put(self.next_obs)


        if self.simulation_complete or not self._init_callback(state_argument):
            return


        with self.act_queue_mutex:
            if self.simulation_complete:
                return
            next_action = self.act_queue.get()
                
        if next_action is None:
            self.simulation_complete = True
            return


        self.last_action = next_action


        if self.rl_option == 1:
            self.cooling_setpoint = next_action
        else:
            if self.baseline_temp == "random":
                self.cooling_setpoint = np.random.randint(20,30)
            else:
                self.cooling_setpoint = self.baseline_temp


        if self.ob_causal == 0 and self.dd_ob_model != None:


            new_row = np.array((Ti, rh, mrt, outdoor_temp, outdoor_rh,
                                direct_solar, diffuse_solar, ground_solar, wind_speed,
                                elec_hvac, self.cooling_setpoint_prev, self.hold_timer_prev,
                                self.hold_status_prev, self.override_prev,
                                time_sin, time_cos,
                                day_of_year_sin, day_of_year_cos,
                                day_of_week_sin, day_of_week_cos)).reshape(1, -1)
            
            new_row[:, :12] = self.scaler.transform(new_row[:, :12])

            
            self.x_input = np.vstack((self.x_input[1:,:], new_row))

        elif self.ob_causal == 4 and self.dd_ob_model != None:

            new_row = np.array((Ti, rh, mrt, self.override_prev)).reshape(1, -1)

            new_row[:, :3] = self.scaler.transform(new_row[:, :3])

            self.x_input = np.vstack((self.x_input[1:,:], new_row))


        self.occupant_simulation(Ti, mrt, rh, time)


        self.x.set_actuator_value(state_argument, self.x.get_actuator_handle(state_argument, "Schedule:Compact", "Schedule Value", "cooling_sch"), self.cooling_setpoint)

        
        self.cooling_setpoint_prev = self.cooling_setpoint
        self.override_prev = self.occupant.override
        self.hold_status_prev = self.hold_status
        self.hold_timer_prev = self.occupant.hold_timer

        r3 = - self.next_obs["elec_hvac"] * tou_rate(time)

        r4 = - self.occupant.override

        r3, r4 = r3/0.14, r4

        reward_array = np.array([r3, r4])
        weighted_reward = reward_array * np.array(self.obj_coeffs)
        reward_sum = np.dot(reward_array, np.array(self.obj_coeffs))
        
        if self.reward_form == "positive":
            reward_sum = reward_sum +1

        if self.save_csv == 1: 
            new_row = pd.DataFrame([{
                'r3': r3,
                'r4': r4,
                'weighted_r3': weighted_reward[0],
                'weighted_r4': weighted_reward[1],
                'reward_sum': reward_sum,
                'time': time,
                'day_of_year': day_of_year,
                'day_of_week': day_of_week,
                'outdoor_temp': outdoor_temp,
                'outdoor_rh': outdoor_rh,
                'direct_solar': direct_solar,
                'diffuse_solar': diffuse_solar,
                'ground_solar': ground_solar,
                'wind_speed': wind_speed,
                'Ti': Ti,
                'rh': rh,
                'mrt': mrt,
                'elec_hvac': elec_hvac,
                'elec_fans_1': elec_fans_1,

                'cooling_capa': cooling_capa,
                'cop': cop,

                'elec_price': tou_rate(time),
                'elec_cost': elec_hvac * tou_rate(time),
                'cooling_setpoint': self.cooling_setpoint,
                'next_action': next_action,
                'dt_obs': self.occupant.dt_obs,
                'override': self.occupant.override,
                'hold_status': self.hold_status,
                'hold_timer': self.occupant.hold_timer,
                'p_override': self.occupant.p_override,
                'term_dist_mu': self.occupant.term_dist_mu,
                'term_acc_disc': self.occupant.term_acc_disc,
                's_0': self.occupant.s_0,
                'dist_mu': self.occupant.dist_mu,
                'acc_disc': self.occupant.acc_disc,
                'on_off_flag': self.on_off_flag,
                'heating_setpoint': self.heating_setpoint
            }])

            self.df = pd.concat([self.df, new_row], ignore_index=True)
        
        if self.save_csv == 1:
            if abs(time - 24) < 0.2 and day_of_year == 243:
                os.makedirs("rl_results/{model_name}".format(model_name = self.model_name), exist_ok=True)
                self.df.to_csv("rl_results/{model_name}/rl_table_ongoing.csv".format(model_name = self.model_name))
                

    def _init_callback(self, state_argument) -> bool:
        """Initialize EnergyPlus handles and checks if simulation runtime is ready."""
        self.initialized = self._init_handles(state_argument) and not self.x.warmup_flag(state_argument)
        return self.initialized

    def _init_handles(self, state_argument):
        """Initialize sensors/actuators handles to interact with during simulation."""
        if not self.initialized:
            if not self.x.api_data_fully_ready(state_argument):
                return False

            self.var_handles = {
                key: self.x.get_variable_handle(state_argument, *var) for key, var in self.variables.items()
            }

            self.meter_handles = {
                key: self.x.get_meter_handle(state_argument, meter) for key, meter in self.meters.items()
            }

            self.actuator_handles = {
                key: self.x.get_actuator_handle(state_argument, *actuator) for key, actuator in self.actuators.items()
            }

            self.x.set_actuator_value(state_argument, self.x.get_actuator_handle(state_argument, "Schedule:Compact", "Schedule Value", "bangbang_sch_heating"), 0)
            self.x.set_actuator_value(state_argument, self.x.get_actuator_handle(state_argument, "Schedule:Compact", "Schedule Value", "heating_sch"), self.heating_setpoint)

            for handles in [self.var_handles, self.meter_handles, self.actuator_handles]:
                if any([v == -1 for v in handles.values()]):
                    available_data = self.x.list_available_api_data_csv(state_argument).decode("utf-8")
                    raise RuntimeError(
                        f"got -1 handle, check your var/meter/actuator names:\n"
                        f"> variables: {self.var_handles}\n"
                        f"> meters: {self.meter_handles}\n"
                        f"> actuators: {self.actuator_handles}\n"
                        f"> available E+ API data: {available_data}"
                    )

            self.initialized = True

        return True

    def _flush_queues(self):
        if self.act_queue.empty():
            self.act_queue.put(None)

        while not self.obs_queue.empty():
            self.obs_queue.get()

        with self.act_queue_mutex:
            while not self.act_queue.empty(): 
                self.act_queue.get()

class EnergyPlusEnv(gym.Env, metaclass=abc.ABCMeta):
    """EnergyPlus gym environment.

    This class implements the gymnasium API. Its ``get_variables`` / ``get_meters`` /
    ``get_actuators`` / ``get_observation_space`` / ``get_action_space`` hooks are
    concrete here, so it is instantiated directly by ``02_controller_training.py``.
    """

    def __init__(self, env_config: Dict[str, Any], 
                weather_file = None,
                 start_month : int = 7, start_day : int = 1, 
                 end_month : int = 7, end_day : int = 1, 
                 rl_option = 1,
                 occ_option = 1,
                 reward_form = "positive", 
                 save_csv = 1,
                 print_values = False, 
                 model_name = "model",
                 obj_coeffs = [1, 1],
                 occ_case = 0,
                 rl_occ_obs = 1,
                 dd_ob_model = None, 
                 scaler = None,
                 baseline_temp = 24,
                 ob_causal = 4,
                 hold_duration = None
                 ):
        self.spec = gym.envs.registration.EnvSpec(f"{self.__class__.__name__}")

        self.env_config = env_config
        self.episode = -1
        self.timestep = 0
        self.rl_occ_obs = rl_occ_obs

        self.observation_space = self.get_observation_space()
        self.last_obs = {}

        self.action_space = self.get_action_space()

        self.default_action = self.post_process_action(12)

        self.energyplus_runner: Optional[EnergyPlusRunner] = None
        self.obs_queue: Optional[Queue] = None
        self.act_queue: Optional[Queue] = None

        self.visualization = None 

        self.start_month = start_month
        self.start_day = start_day
        self.end_month = end_month
        self.end_day = end_day

        raw_file = str(IDF_DIR / "00_building_model.idf")

        if not os.path.exists(os.path.join(str(IDF_DIR), "00_building_model_{}_{}_{}_{}.idf").format(start_month, start_day, end_month, end_day)):
            file_process = IDF_handler(raw_file)
            file_process.edit_and_save(start_month=self.start_month, start_day=self.start_day, end_month=self.end_month, end_day=self.end_day)

        self.rl_option = rl_option
        self.occ_option = occ_option
        self.save_csv = save_csv
        if weather_file is None:
            weather_file = pd.read_csv(DATA_DIR / "weather_toronto.csv")
        self.weather_file = weather_file
        self.model_name = model_name
        self.obj_coeffs = obj_coeffs
        self.reward_form = reward_form
        self.baseline_temp = baseline_temp


        self.print_values = print_values
        self.runner_config = RunnerConfig(
            epw=self.get_weather_file(),
            idf=self.get_idf_file(),
            output=self.env_config["output"],
            variables=self.get_variables(),
            meters=self.get_meters(),
            actuators=self.get_actuators(),
            csv=self.env_config.get("csv", True),
            verbose=self.env_config.get("verbose", True),
        )

        self.occ_case = occ_case
        self.dd_ob_model = dd_ob_model
        self.scaler = scaler
        self.ob_causal = ob_causal
        self.hold_duration = hold_duration
    def get_weather_file(self) -> Union[Path, str]:
        """Returns the path to a valid weather file (.epw).

        Called once from ``__init__`` when the runner configuration is built.
        """
        return str(DATA_DIR / "toronto.epw")

    def get_idf_file(self) -> Union[Path, str]:
        """Returns the path to a valid IDF file."""
        new_file_dir = os.path.join(str(IDF_DIR), "00_building_model_{}_{}_{}_{}.idf").format(self.start_month, self.start_day, self.end_month, self.end_day)
        return new_file_dir

    def get_observation_space(self) -> gym.Space:
        """Returns the observation space of the environment."""
        num_val = 31+1

        if self.rl_occ_obs == 0:
            num_val = 28+1 

        return spaces.Box(low=-np.ones(num_val), high=np.ones(num_val), shape = (num_val,), dtype=np.float32)

    def get_action_space(self) -> gym.Space:
        """Returns the action space of the environment."""
        n_set_ranges = 21
        return spaces.Discrete(n_set_ranges)
    
    def computing_reward(self, obs: Dict[str, float]) -> Tuple[float, float, float]:
        """Computes the reward for the given observation.

        Returns ``(reward_sum, weighted_r3, weighted_r4)``: the scalar reward plus its
        weighted electricity-cost and override components.
        """
        time_sin = obs["time_sin"]
        time_cos = obs["time_cos"]
        time = np.arctan2(time_sin, time_cos) * 12 / np.pi

        if time < 0:
            time += 24

        r3 = - obs["elec_hvac"] * tou_rate(time) 

        if self.rl_occ_obs == 0:
            r4  = 0
        else:
            r4 = - obs["occupant_override"] 

        r3, r4 = r3/0.14, r4


        reward_array = np.array([r3, r4])

        coefficients = np.array(self.obj_coeffs)

        reward_sum = np.dot(reward_array, coefficients)
        if self.reward_form == "positive":
            reward_sum = reward_sum +1

        return reward_sum, r3*coefficients[0], r4*coefficients[1]

    def get_variables(self) -> Dict[str, Tuple[str, str]]:
        """Returns the variables to track during simulation."""
        var_list = {
            "Ti": ("Zone Mean Air Temperature", "living_unit1"),
            "mrt": ("Zone Mean Radiant Temperature", "living_unit1"),
            "rh": ("Zone Air Relative Humidity", "living_unit1"),
            "outdoor_temp": ("Site Outdoor Air Drybulb Temperature", "Environment"),
            "outdoor_rh": ("Site Outdoor Air Relative Humidity", "Environment"),
            "direct_solar": ("Site Direct Solar Radiation Rate per Area", "Environment"),
            "diffuse_solar": ("Site Diffuse Solar Radiation Rate per Area", "Environment"),
            "ground_solar": ("Site Ground Reflected Solar Radiation Rate per Area", "Environment"),
            "wind_speed": ("Site Wind Speed", "Environment")}
        return var_list

    def get_meters(self) -> Dict[str, str]:
        met_list =  {"elec_hvac": "Electricity:HVAC"}
        """Returns the meters to track during simulation."""
        return met_list


    def get_actuators(self) -> Dict[str, Tuple[str, str, str]]:
        act_list = {"bangbang" : ("Schedule:Compact", "Schedule Value", "bangbang_sch")}
        """Returns the actuators to control during simulation."""
        return act_list

    def post_process_action(self, action: Union[float, List[float]]) -> Union[float, List[float]]:
        """Post-processes the action(s) before sending it to EnergyPlus."""
        

        action = action * 0.5 + 20
        return action

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)

        if self.energyplus_runner is not None:
            self.energyplus_runner.stop()

        self.episode += 1
        self.last_obs = self.observation_space.sample()
       
        self.obs_queue = Queue(maxsize=1)
        self.act_queue = Queue(maxsize=1)
       
        self.energyplus_runner = EnergyPlusRunner(
            episode=self.episode,
            obs_queue=self.obs_queue,
            act_queue=self.act_queue,
            runner_config=self.runner_config,
            rl_option = self.rl_option,
            occ_option = self.occ_option,
            save_csv= self.save_csv,
            weather_file=self.weather_file,
            model_name = self.model_name,
            obj_coeffs = self.obj_coeffs,
            reward_form = self.reward_form,
            occ_case = self.occ_case,
            rl_occ_obs = self.rl_occ_obs,
            baseline_temp = self.baseline_temp,
            dd_ob_model = self.dd_ob_model,
            scaler = self.scaler,
            ob_causal = self.ob_causal,
            hold_duration = self.hold_duration
        )
        self.energyplus_runner.start()
        
        self.last_obs = obs = self.energyplus_runner.init_exchange(default_action=self.default_action)

        try:
            return np.array(list(obs.values()), dtype=np.float32), {}
        
        except Exception as e:
            print(f"Error in reset: {e} - seems like energyplus does not work properly.")
            os.kill(os.getpid(), signal.SIGTERM)


    def step(self, action):
        self.timestep += 1

        done = False

        try:
            if self.energyplus_runner.failed():
                raise RuntimeError(f"EnergyPlus failed with {self.energyplus_runner.sim_results['exit_code']}")
        
        except Exception as e:
            print(f"Simulation error: {e}")
            os.kill(os.getpid(), signal.SIGTERM)
   
        if self.energyplus_runner.simulation_complete:
            done = True
            obs = self.last_obs
            self.energyplus_runner.stop()
        else:
            action_to_apply = self.post_process_action(action)
            timeout = 2
            try:
                self.act_queue.put(action_to_apply, timeout=timeout)
                obs = self.obs_queue.get(timeout=timeout)
            except (Full, Empty):
                obs = None
                pass
            
            if obs is None:
                done = True
                obs = self.last_obs
                self.energyplus_runner.stop()
            else:
                self.last_obs = obs

        reward, r3, r4 = self.computing_reward(obs)


        if self.print_values == 1:
            pass
            print("Ti", obs["Ti"], "reward", reward, {"r3": r3, "r4" : r4, "Ti": obs["Ti"]},"done", done, "action", action_to_apply)
        
        obs_vec = np.array(list(obs.values()), dtype=np.float32)
        
        return obs_vec, reward, done, False, {"r3": r3, "r4": r4, "Ti": obs["Ti"]}

    def close(self):
        self.energyplus_runner.stop()

        if self.energyplus_runner is not None:
            self.energyplus_runner.stop()

    def render(self):
        return 

def tou_rate(time):
    off_peak = 0.074
    mid_peak = 0.102
    on_peak = 0.151

    if 0 <= time < 7:
        return off_peak
    elif time <11:
        return mid_peak
    elif time < 17:
        return on_peak
    elif time < 19:
        return mid_peak
    else:
        return off_peak
    

def thermostat_activity(time):

    if 0 <= time < 6.5:
        return 0 
    elif 6.5 <= time < 23.5:
        return 1
    elif 23.5 <= time:
        return 0
