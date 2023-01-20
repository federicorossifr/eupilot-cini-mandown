'''Memorization Utils'''

import time
import math
import pynvml

def test_GPU():

    pynvml.nvmlInit()  # initialize NVIDIA Management Library (NVML)
    index = pynvml.nvmlDeviceGetHandleByIndex(0)
    NVML_TEMPERATURE_GPU = 0

    # Conversion factors:
    B_to_MiB = 1048576  # conversion factor from Bytes to Mebibyte (2^20)
    mW_to_W = 1000  # conversion factor from milliWatt to Watt 

    GPU_name = str(pynvml.nvmlDeviceGetName(index)).replace("b", "")  # get GPU name
    GPU_id = pynvml.nvmlDeviceGetIndex(index)  # get GPU ID
    GPU_total_memory = pynvml.nvmlDeviceGetMemoryInfo(index).total/B_to_MiB  # get total GPU memory in MebiBytes
    GPU_used_memory = pynvml.nvmlDeviceGetMemoryInfo(index).used/B_to_MiB  # get used GPU memory in MebiBytes
    GPU_free_memory = pynvml.nvmlDeviceGetMemoryInfo(index).free/B_to_MiB  # get free GPU memory in MebiBytes
    GPU_utilization_rate = pynvml.nvmlDeviceGetUtilizationRates(index).gpu  # get GPU utilization rate in %

    print(f"""GPU properties:
    Name: {GPU_name}
    ID: {GPU_id}
    Total Memory: {GPU_total_memory:.0f} MiB
    Used Memory: {GPU_used_memory} MiB
    Free Memory: {GPU_free_memory} MiB
    Utilization Rate: {GPU_utilization_rate} % """)
    print("---------------------------------------------------------------------------------------")
    
    N_measures = 500
    T_meas = []
    P_meas = []
    x0 = []
    x2 = []
    
    i = 0

    while(i < N_measures):

        print("Measure Number: ", i+1)

        GPU_temperature = pynvml.nvmlDeviceGetTemperature(index, NVML_TEMPERATURE_GPU)  # get GPU temperature in °C
        GPU_power_consumption = pynvml.nvmlDeviceGetPowerUsage(index)/mW_to_W  # get GPU power consumption in W
        print(f"""Temperature": {GPU_temperature} °C""")
        print(f"""Power Consumption: {GPU_power_consumption:.1f} W""")

        # Temperature
        T_meas.append(GPU_temperature)
        N_Temp = len(T_meas)
        sum_Temp = sum(T_meas)
        mean_Temp = sum_Temp/N_Temp
        T_max = max(T_meas)
        T_min = min(T_meas)
        print(T_meas)
        print("Temperature Mean: ", mean_Temp)
        print("Max Temperature: ", T_max)
        print("Min Temperature: ", T_min)

        x0.append((GPU_temperature - mean_Temp)*(GPU_temperature - mean_Temp))
        x1 = sum(x0)
        variance_Temp = x1/N_Temp
        std_dev_Temp = math.sqrt(variance_Temp)
        print("Temperature Variance: ", variance_Temp)
        print("Temperature Standard Deviation: ", std_dev_Temp)

        # Power Consumption
        P_meas.append(GPU_power_consumption)
        N_Power = len(P_meas)
        sum_Power = sum(P_meas)
        mean_Power = sum_Power/N_Power
        P_max = max(P_meas)
        P_min = min(P_meas)
        print(P_meas)
        print("Power Consumption Mean: ", mean_Power)
        print("Max Power Consumption: ", P_max)
        print("Min Power Consumption: ", P_min)

        x2.append((GPU_power_consumption - mean_Power)*(GPU_power_consumption - mean_Power))
        x3 = sum(x2)
        variance_Power = x3/N_Power
        std_dev_Power = math.sqrt(variance_Power)
        print("Power Consumption Variance: ", variance_Power)
        print("Power Consumption Standard Deviation: ", std_dev_Power)

        i = i+1
        time.sleep(3)

        print("---------------------------------------------------------------------------------------")

class SaveData:

    def __init__(self, save_dir, device):

        self.test_info = 0
        self.save_dir = save_dir

        if str(device) != 'cpu':

            pynvml.nvmlInit()  # initialize NVIDIA Management Library (NVML)
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.B_to_MiB = 1048576  # conversion factor from Bytes to Mebibyte (2^20)
            self.NVML_TEMPERATURE_GPU = 0

            GPU_name = str(pynvml.nvmlDeviceGetName(self.handle)).replace("b", "")  # get GPU name
            GPU_id = pynvml.nvmlDeviceGetIndex(self.handle)  # get GPU ID
            GPU_total_memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle).total  # get total GPU memory in Bytes
            GPU_temperature = pynvml.nvmlDeviceGetTemperature(self.handle, self.NVML_TEMPERATURE_GPU)  # get GPU temperature in °C
            GPU_power_consumption = pynvml.nvmlDeviceGetPowerUsage(self.handle)/1000  # get GPU power consumption in W

            print(f"""GPU properties:
            Name: {GPU_name}
            Device Index: {GPU_id}
            Total Memory: {GPU_total_memory/self.B_to_MiB:.0f} MiB
            Temperature: {GPU_temperature} °C
            Power Consumption: {GPU_power_consumption:.1f} W""")

    def get_GPU_info(self):

        GPU_memory_used = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used  # get total GPU used in Bytes
        GPU_utilization_rate = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu 
        memory_utilization_rate = pynvml.nvmlDeviceGetUtilizationRates(self.handle).memory
        GPU_temperature = pynvml.nvmlDeviceGetTemperature(self.handle, self.NVML_TEMPERATURE_GPU)  # get GPU temperature in °C
        GPU_power_consumption = pynvml.nvmlDeviceGetPowerUsage(self.handle)/1000  # get GPU power consumption in W

        GPU_info = [GPU_memory_used, GPU_utilization_rate, memory_utilization_rate, GPU_temperature, GPU_power_consumption]

        return GPU_info

    def get_speed_info(self, dt):

        pre_process_speed = dt[0]*1000  # pre-process speed (in ms)
        inference_speed = dt[1]*1000  # inference speed (in ms)
        nms_speed = dt[2]*1000  # NMS speed (in ms)
        man_down_speed = dt[3]*1000  # man down classifier speed (in ms)
        deep_sort_speed = dt[4]*1000  # DeepSORT algorithm speed (in ms)

        speed_info = [pre_process_speed, inference_speed, nms_speed, man_down_speed, deep_sort_speed]

        return speed_info

    def save(self, speed_info, GPU_info = None):
        
        x1 = speed_info
        if GPU_info is not None:
            x2 = GPU_info

        with open(self.save_dir / 'test_info.txt', 'a') as f:
            if self.test_info == 0:
                f.truncate(0)
                self.test_info = 1
            f.write(str(x1[0]) + ' ' + str(x1[1]) + ' ' + str(x1[2]) + ' ' + str(x1[3]) + ' ' + str(x1[4]))
            if GPU_info is not None:
                f.write(' ' + str(x2[0]) + ' ' + str(x2[1]) + ' ' + str(x2[2]) + ' ' + str(x2[3]) + ' ' + str(x2[4]))
            f.write('\n')