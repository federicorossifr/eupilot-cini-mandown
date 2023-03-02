# Man Down Tracking 🚀

import platform
import time
import math
import torch
import psutil

from tools.general import color_str

if torch.cuda.is_available():
    import pynvml

def get_CPU_informations():

    CPU_name = platform.processor()
    CPU_architecture = platform.machine()
    CPU_core_number = psutil.cpu_count()
    CPU_frequency = psutil.cpu_freq().current
    print(f"{color_str('bold', 'blue', 'CPU properties:')} Name: {CPU_name} | Architecture: {CPU_architecture} | Core Number: {CPU_core_number} | Frequency: {CPU_frequency} MHz \n")    

    U, T, x0, i, N = [], [], [], 0, 500

    while(i < N):

        print(f"{color_str('bold', 'white', 'Measure Number:')} {i+1}")

        # Utilization Rate:
        utilization_rate = psutil.cpu_percent()  # get CPU utilization rate in %
        U.append(utilization_rate)
        utilization_rate_mean = sum(U)/len(U)
        print(f'Utilization Rate: {utilization_rate} %')
        print(f'Utilization Rate Mean: {utilization_rate_mean} %')

        if platform.system() != 'Windows' and psutil.sensors_temperatures() is not None:

            # Temperature:
            temperature = psutil.sensors_temperatures()  # get CPU temperature in °C
            T.append(temperature)
            temperature_mean = sum(T)/len(T)
            x0.append((temperature - temperature_mean)*(temperature - temperature_mean))
            x1 = sum(x0)
            temperature_variance = x1/len(T)
            temperature_std_dev = math.sqrt(temperature_variance)
            print(f'Temperature: {temperature} °C')
            print(f'Temperature Mean: {temperature_mean} °C')
            print(f'Temperature Standard Deviation: {temperature_std_dev} °C')

        print('\n')

        i += 1
        time.sleep(3)

def get_GPU_informations():

    if torch.cuda.is_available():
        pynvml.nvmlInit()  # initialize NVIDIA Management Library (NVML)
        index = pynvml.nvmlDeviceGetHandleByIndex(0)

        B_to_MiB = 1048576  # conversion factor from Bytes to Mebibyte (2^20)
        mW_to_W = 1000  # conversion factor from milliWatt to Watt 

        GPU_name = str(pynvml.nvmlDeviceGetName(index)).replace("b", "")  # get GPU name
        GPU_id = pynvml.nvmlDeviceGetIndex(index)  # get GPU ID
        GPU_total_memory = pynvml.nvmlDeviceGetMemoryInfo(index).total/B_to_MiB  # get total GPU memory in MebiBytes
        GPU_used_memory = pynvml.nvmlDeviceGetMemoryInfo(index).used/B_to_MiB  # get used GPU memory in MebiBytes
        GPU_free_memory = pynvml.nvmlDeviceGetMemoryInfo(index).free/B_to_MiB  # get free GPU memory in MebiBytes
        GPU_utilization_rate = pynvml.nvmlDeviceGetUtilizationRates(index).gpu  # get GPU utilization rate in %

        print(f"{color_str('bold', 'blue', 'GPU properties:')} Name: {GPU_name} | Device Index: {GPU_id} | Total Memory: {GPU_total_memory:.0f} MiB | Utilization Rate: {GPU_utilization_rate} %\n")

        T, PC, x0, x2, i, N = [], [], [], [], 0, 500

        while(i < N):

            print(f"{color_str('bold', 'white', 'Measure Number:')} {i+1}")

            # Temperature:
            temperature = pynvml.nvmlDeviceGetTemperature(index, 0)  # get GPU temperature in °C
            T.append(temperature)
            temperature_mean = sum(T)/len(T)
            x0.append((temperature - temperature_mean)*(temperature - temperature_mean))
            x1 = sum(x0)
            temperature_variance = x1/len(T)
            temperature_std_dev = math.sqrt(temperature_variance)
            print(f'Temperature: {temperature} °C')
            print(f'Temperature Mean: {temperature_mean} °C')
            print(f'Temperature Standard Deviation: {temperature_std_dev} °C')

            # Power Consumption:
            power_consumption = pynvml.nvmlDeviceGetPowerUsage(index)/mW_to_W  # get GPU power consumption in W
            PC.append(power_consumption)
            power_consumption_mean = sum(PC)/len(PC)
            x2.append((power_consumption - power_consumption_mean)*(power_consumption - power_consumption_mean))
            x3 = sum(x2)
            power_consumption_variance = x3/len(PC)
            power_consumption_std_dev = math.sqrt(power_consumption_variance)
            print(f'Power Consumption: {power_consumption} W')
            print(f'Power Consumption Mean: {power_consumption_mean} W')
            print(f'Power Consumption Standard Deviation: {power_consumption_std_dev} W \n')

            i += 1
            time.sleep(3)

class SaveInfo:

    def __init__(self, save_dir, device = 'cpu'):

        self.write_info = True
        self.save_dir = save_dir
        self.device = device

        CPU_name = platform.processor()
        CPU_architecture = platform.machine()
        CPU_utilization_rate = psutil.cpu_percent()  # get CPU utilization rate in %
        CPU_temperature = 0.0  # get CPU temperature in °C
        CPU_power_consumption = 0.0  # get CPU power consumption in W

        print(f"{color_str('bold', 'red', 'CPU properties:')} Name: {CPU_name} | Architecture: {CPU_architecture} | Utilization Rate: {CPU_utilization_rate} % | Temperature: {CPU_temperature} °C | Power Consumption: {CPU_power_consumption} W")

        if str(self.device) != 'cpu':
            pynvml.nvmlInit()  # initialize NVIDIA Management Library (NVML)
            self.index = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.B_to_MiB = 1048576  # conversion factor from Bytes to Mebibyte (2^20)
            self.NVML_TEMPERATURE_GPU = 0

            GPU_name = str(pynvml.nvmlDeviceGetName(self.index)).replace("b", "")  # get GPU name
            GPU_id = pynvml.nvmlDeviceGetIndex(self.index)  # get GPU ID
            GPU_total_memory = pynvml.nvmlDeviceGetMemoryInfo(self.index).total/self.B_to_MiB  # get total GPU memory in MiB
            GPU_temperature = pynvml.nvmlDeviceGetTemperature(self.index, self.NVML_TEMPERATURE_GPU)  # get GPU temperature in °C
            GPU_power_consumption = pynvml.nvmlDeviceGetPowerUsage(self.index)/1000  # get GPU power consumption in W

            print(f"{color_str('bold', 'blue', 'GPU properties:')} Name: {GPU_name} | Device Index: {GPU_id} | Total Memory: {GPU_total_memory:.0f} MiB | Temperature: {GPU_temperature} °C | Power Consumption: {GPU_power_consumption:.1f} W")

    def get_speed_informations(self, dt):

        pre_process_speed = dt[0]*1000  # pre-process speed (in ms)
        inference_speed = dt[1]*1000  # inference speed (in ms)
        post_process_speed = dt[2]*1000  # post-process speed (in ms)
        man_down_speed = dt[3]*1000  # man down classifier speed (in ms)
        deep_sort_speed = dt[4]*1000  # DeepSORT algorithm speed (in ms)

        speed_info = [pre_process_speed, inference_speed, post_process_speed, man_down_speed, deep_sort_speed]

        return speed_info

    def get_CPU_informations(self):
        
        CPU_utilization_rate = psutil.cpu_percent()  # get CPU utilization rate in %
        CPU_temperature = 0.0  # get CPU temperature in °C
        CPU_power_consumption = 0.0  # get CPU power consumption in W

        CPU_info = [CPU_utilization_rate, CPU_temperature, CPU_power_consumption]

        return CPU_info

    def get_GPU_informations(self):

        GPU_memory_used = pynvml.nvmlDeviceGetMemoryInfo(self.index).used/self.B_to_MiB  # get GPU memory used in MiB
        GPU_utilization_rate = pynvml.nvmlDeviceGetUtilizationRates(self.index).gpu  # get GPU utilization rate in %
        GPU_temperature = pynvml.nvmlDeviceGetTemperature(self.index, self.NVML_TEMPERATURE_GPU)  # get GPU temperature in °C
        GPU_power_consumption = pynvml.nvmlDeviceGetPowerUsage(self.index)/1000  # get GPU power consumption in W

        GPU_info = [GPU_memory_used, GPU_utilization_rate, GPU_temperature, GPU_power_consumption]

        return GPU_info

    def save(self, speed_info, CPU_info, GPU_info = None):

        with open(self.save_dir / 'info.txt', 'a') as f:
            if self.write_info:
                f.truncate(0)
                self.write_info = False

            # Write speed informations:
            f.write(str(speed_info[0]) + ' ' + str(speed_info[1]) + ' ' + str(speed_info[2]) + 
                ' ' + str(speed_info[3]) + ' ' + str(speed_info[4]))
            
            # Write CPU informations:
            f.write(' ' + str(CPU_info[0]) + ' ' + str(CPU_info[1]) + ' ' + str(CPU_info[2]))

            # Write GPU informations (if available):
            if GPU_info is not None:
                f.write(' ' + str(GPU_info[0]) + ' ' + str(GPU_info[1]) + ' ' + str(GPU_info[2]) + ' ' + str(GPU_info[3]))

            f.write('\n')