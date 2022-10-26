# sitting-posture-detection
sitting posture detection device with arduino 、python
嵌入式部分采用arduino通过两个多路复用器和柔性压力传感器阵列连接，一个提供一侧扫描的电压，一个读取另一侧的电压，读取的数据分十级通过蓝牙发送到PC，进一步可视化和分析。可视化部分用opencv，分析部分采用Mobilenet
V3对采集的压力分布图分类，得到相应的结果。
