'''Memorization Utils'''

import pynvml

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

        speed = dt*1000
        pre_process_speed = speed[0]  # pre-process speed in ms
        inference_speed = speed[1]  # inference speed in ms
        nms_speed = speed[2]  # NMS speed in ms
        man_down_speed = speed[3]  # Man Down speed in ms
        deep_sort_speed = speed[4]  # Deep Sort speed in ms

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