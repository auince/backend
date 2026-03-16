
import math
from pydantic import BaseModel, Field

def calculate_target_geolocation(plane_lon, plane_lat, plane_alt, 
                               plane_heading, plane_pitch, plane_roll,
                               payload_azimuth, payload_pitch):
    """
    根据无人机和载荷参数计算目标的地理坐标
    (Copied from TargetLocate.py)
    """
    
    # 计算载荷相对于地理坐标系的绝对角度
    absolute_azimuth = plane_heading + payload_azimuth
    absolute_pitch = plane_pitch + payload_pitch
    
    # 规范化角度到[-180, 180]
    while absolute_azimuth > 180:
        absolute_azimuth -= 360
    while absolute_azimuth < -180:
        absolute_azimuth += 360
    
    # 防止俯仰角为0导致除零错误
    if abs(absolute_pitch) < 0.001:
        absolute_pitch = -0.001
    
    # 计算目标距离和水平距离
    target_slant_range = abs(plane_alt / math.sin(math.radians(absolute_pitch)))
    horizontal_distance = target_slant_range * math.cos(math.radians(absolute_pitch))
    
    # 计算目标相对于飞机的北向和东向距离
    delta_north = horizontal_distance * math.cos(math.radians(absolute_azimuth))
    delta_east = horizontal_distance * math.sin(math.radians(absolute_azimuth))
    
    # 地球半径(米)
    earth_radius = 6378137
    
    # 计算目标点的经纬度
    # 纬度变化(度)
    delta_lat = math.degrees(delta_north / earth_radius)
    
    # 经度变化(度) - 需要考虑纬度对经度变化的影响
    delta_lon = math.degrees(delta_east / (earth_radius * math.cos(math.radians(plane_lat))))
    
    # 目标经纬度
    target_lat = plane_lat + delta_lat
    target_lon = plane_lon + delta_lon
    
    return target_lon, target_lat

def decimal_to_dms(decimal_degrees, is_latitude=True):
    """
    将十进制经纬度转换为度分秒格式
    (Copied logic from Det.py)
    """
    is_negative = decimal_degrees < 0
    decimal_degrees = abs(decimal_degrees)
    
    degrees = int(decimal_degrees)
    minutes = int((decimal_degrees - degrees) * 60)
    seconds = (decimal_degrees - degrees - minutes/60) * 3600
    
    direction = ''
    if is_latitude:
        direction = 'N' if not is_negative else 'S'
    else:
        direction = 'E' if not is_negative else 'W'
        
    return f"{degrees}°{minutes:02d}'{seconds:.2f}\"{direction}"
