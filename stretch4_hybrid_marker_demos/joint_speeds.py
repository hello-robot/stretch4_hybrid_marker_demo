joint_speeds_dict = {
    'omnibase': {
        'default': {
            'accel_w_r': 1.0,
            'vel_w_r': 1.0,
            'accel_xy_m': 0.5,
            'vel_xy_m': 0.8},
        'fast': {
            'accel_w_r': 2.0,
            'vel_w_r': 2.0,
            'accel_xy_m': 0.3,
            'vel_xy_m': 0.3},
        'max': {
            'accel_w_r': 4.0,
            'vel_w_r': 4.0,
            'accel_xy_m': 0.3,
            'vel_xy_m': 0.3},
        'slow': {
            'accel_w_r': 1.0,
            'vel_w_r': 0.5,
            'accel_xy_m': 0.1,
            'vel_xy_m': 0.1}
        },
    'lift': {
        'default':{
            'accel_m': 0.3,
            'vel_m': 0.3},
        'fast':{
            'accel_m': 0.5,
            'vel_m': 0.4},
        'max':{
            'accel_m': 1.0,
            'vel_m': 0.5},
        'slow':{
            'accel_m': 0.2,
              'vel_m': 0.15}
          }, 
    'arm':{
        'default':{
            'accel_m': 0.4,
            'vel_m': 0.4},
        'fast':{
            'accel_m': 0.6,
            'vel_m': 0.6},
        'max':{
            'accel_m': 0.7,
            'vel_m': 0.7},
        'slow':{
            'accel_m': 0.1,
            'vel_m': 0.1}
    },
    'gripper': {
        'default': {
            'accel': 6.0, 
            'vel': 6.0
        },
        'fast': {
            'accel': 6.0, 
            'vel': 6.0
        },
        'max': {
            'accel': 6.0, 
            'vel': 6.0
        },
        'slow': {
            'accel': 4.0, 
            'vel': 1.0
        }
    },
    'wrist_yaw': {
        'default': {
            'accel': 7.0, 
            'vel': 7.0
        },
        'fast': {
            'accel': 9.0, 
            'vel': 9.0
        },
        'max': {
            'accel': 12.0, 
            'vel': 12.0
        },
        'slow': {
            'accel': 4.0, 
            'vel': 4.0
        }
    },
    'wrist_pitch': {
        'default': {
            'accel': 7.0, 
            'vel': 7.0
        },
        'fast': {
            'accel': 9.0, 
            'vel': 9.0
        },
        'max': {
            'accel': 12.0, 
            'vel': 12.0
        },
        'slow': {
            'accel': 4.0, 
            'vel': 4.0
        }
    },
    'wrist_roll': {
        'default': {
            'accel': 7.0, 
            'vel': 7.0
        },
        'fast': {
            'accel': 9.0, 
            'vel': 9.0
        },
        'max': {
            'accel': 12.0, 
            'vel': 12.0
        },
        'slow': {
            'accel': 4.0, 
            'vel': 4.0
        }
    }
}