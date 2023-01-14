weight_sr_l1 = 1  # 0.8
weight_speed = 0.1

mobile_device='huawei_p30'
compute_device='cpu'


def update_weight(params):
    params.weight_sr_l1 = weight_sr_l1
    params.weight_speed = weight_speed
