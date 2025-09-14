# RL Based Thermal Comfort Controller 🌡️🤖

## 🎯 Purpose
The goal of this project is to **design and implement a Reinforcement Learning (RL)-based controller** that maintains indoor thermal comfort while optimizing energy consumption.  
Traditional HVAC controllers (like thermostats or PID) are simple but limited — they cannot anticipate setpoint changes, dynamic prices, or real-world uncertainties.  
Our RL controller aims to:
- Keep indoor temperature within the comfort band.  
- Reduce energy usage and avoid peak demand.  
- Remain robust under noise, delays, and capacity changes.  

To validate the approach, we will build a **scaled demo house model** with sensors, actuators, and a microcontroller running the RL-based strategy.

## 🛠️ Demo Model & Hardware
The prototype setup will include:
- **Microcontroller / Edge Device** (Arduino, ESP32, or Raspberry Pi) to run the controller.  
- **Sensors**:  
  - Temperature & Humidity sensors (e.g., DHT22, DS18B20)  
  - Environmental sensors for noise, light, or airflow (optional)  
- **Actuators**:  
  - Heating/Cooling element (resistive heater, fan, or Peltier module)  
  - Ventilation fan with adjustable speed (for airflow control)  
- **House Model**:  
  - Scaled demo enclosure representing a thermal zone  
  - Windows/doors for disturbance simulation  
- **IoT Connectivity** (ESP8266/ESP32) for real-time monitoring and logging.

## 📂 Repository Contents
- `Report/` → Project documentation and technical notes  
- `Code/` → RL algorithms (PPO/SAC) and controller implementation  
- `Circuits/` → Circuit diagrams and hardware schematics  
- `Model/` → Demo house model and hardware setup details  

---

This project demonstrates how **machine learning and smart sensing** can improve comfort, energy efficiency, and resilience of building climate control systems.
