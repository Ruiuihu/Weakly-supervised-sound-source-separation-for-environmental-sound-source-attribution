#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频频谱图生成脚本 - ESR项目
功能：将output_slices_ESR目录中的所有WAV文件转换为频谱图
作者：AI Assistant
日期：2024
"""

import os
import librosa
from utils_t import amplitude, Afilter
from scipy.signal import stft
import numpy as np
import soundfile as sf
from matplotlib.image import imsave
from pathlib import Path
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('spectrogram_generation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SpectrogramGenerator:
    def __init__(self, target_amplitude=60, nperseg=512):
        """
        频谱图生成器初始化
        
        Args:
            target_amplitude (int): 目标幅度，默认60
            nperseg (int): STFT窗口长度，默认512
        """
        self.target_amplitude = target_amplitude
        self.nperseg = nperseg
    
    def load_audio(self, file_path):
        """
        加载音频文件
        
        Args:
            file_path (str): 音频文件路径
            
        Returns:
            tuple: (audio_data, sample_rate) 或 (None, None) 如果加载失败
        """
        try:
            sound, sr = sf.read(file_path)
            # 如果是立体声，转换为单声道
            if len(sound.shape) > 1:
                sound = sound[:, 0]
            return sound, sr
        except Exception as e:
            logger.error(f"加载音频失败 {file_path}: {str(e)}")
            return None, None
    
    def generate_spectrogram(self, sound, sr):
        """
        生成频谱图数据
        
        Args:
            sound (np.array): 音频数据
            sr (int): 采样率
            
        Returns:
            np.array: 频谱图数据，如果生成失败返回None
        """
        try:
            # 应用幅度调整
            sound = amplitude(sound, sr, self.target_amplitude)
            
            # 应用A滤波器
            filtered_sound = Afilter(sound, sr)
            
            # 计算STFT
            _, _, stft_data = stft(filtered_sound, sr, nperseg=self.nperseg)
            
            # 转换为分贝标度
            stft_data = librosa.amplitude_to_db(np.abs(stft_data), ref=np.max)
            
            # 翻转频谱图（低频在下，高频在上）
            stft_data = np.flipud(stft_data)
            
            return stft_data
        except Exception as e:
            logger.error(f"生成频谱图失败: {str(e)}")
            return None
    
    def save_spectrogram(self, spectrogram_data, output_path):
        """
        保存频谱图为图像文件
        
        Args:
            spectrogram_data (np.array): 频谱图数据
            output_path (str): 输出文件路径（.jpg格式）
            
        Returns:
            bool: 保存是否成功
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存频谱图为JPG文件
            imsave(output_path, spectrogram_data, cmap='hot')
            return True
        except Exception as e:
            logger.error(f"保存频谱图失败 {output_path}: {str(e)}")
            return False
    
    def process_single_file(self, input_file, output_file):
        """
        处理单个音频文件，生成频谱图
        
        Args:
            input_file (str): 输入音频文件路径
            output_file (str): 输出频谱图文件路径
            
        Returns:
            bool: 处理是否成功
        """
        # 加载音频
        sound, sr = self.load_audio(input_file)
        if sound is None:
            return False
        
        # 生成频谱图
        spectrogram_data = self.generate_spectrogram(sound, sr)
        if spectrogram_data is None:
            return False
        
        # 保存频谱图
        return self.save_spectrogram(spectrogram_data, output_file)
    
    def get_output_path(self, input_file, input_root, output_root):
        """
        根据输入文件路径生成对应的输出文件路径，保持目录结构
        
        Args:
            input_file (str): 输入文件路径
            input_root (str): 输入根目录
            output_root (str): 输出根目录
            
        Returns:
            str: 输出文件路径（.jpg格式）
        """
        # 计算相对路径
        input_path = Path(input_file)
        input_root_path = Path(input_root)
        
        # 获取相对于输入根目录的路径
        relative_path = input_path.relative_to(input_root_path)
        
        # 将文件扩展名改为.jpg
        relative_path = relative_path.with_suffix('.jpg')
        
        # 生成输出路径
        output_path = Path(output_root) / relative_path
        
        return str(output_path)
    
    def process_directory(self, input_dir, output_dir):
        """
        递归处理目录下的所有WAV文件，生成频谱图
        
        Args:
            input_dir (str): 输入目录路径
            output_dir (str): 输出目录路径
        """
        # 查找所有WAV文件
        input_path = Path(input_dir)
        wav_files = list(input_path.rglob("*.wav")) + list(input_path.rglob("*.WAV"))
        
        if not wav_files:
            logger.warning(f"在目录 {input_dir} 中未找到WAV文件")
            return
        
        # 处理每个文件
        success_count = 0
        failed_count = 0
        
        for wav_file in tqdm(wav_files, desc="生成频谱图"):
            try:
                # 生成输出文件路径
                output_file = self.get_output_path(str(wav_file), input_dir, output_dir)
                
                # 处理文件
                if self.process_single_file(str(wav_file), output_file):
                    success_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"处理文件时出错 {wav_file}: {str(e)}")
                failed_count += 1
        
        logger.info(f"频谱图生成完成! 成功: {success_count}, 失败: {failed_count}")


def main():
    """
    主函数
    """
    # 配置参数
    INPUT_DIR = "output_slices_province"  # 输入目录（音频切片目录）
    OUTPUT_DIR = "spectrograms_province"  # 输出目录（频谱图目录）
    
    # 检查输入目录是否存在
    if not os.path.exists(INPUT_DIR):
        logger.error(f"输入目录不存在: {INPUT_DIR}")
        return
    
    # 创建频谱图生成器
    generator = SpectrogramGenerator(
        target_amplitude=60,  # 目标幅度60dB
        nperseg=512          # STFT窗口长度512
    )
    
    # 开始处理
    try:
        generator.process_directory(INPUT_DIR, OUTPUT_DIR)
        logger.info("所有任务完成!")
    except KeyboardInterrupt:
        logger.info("用户中断处理")
    except Exception as e:
        logger.error(f"处理过程中出现错误: {str(e)}")


if __name__ == "__main__":
    main() 