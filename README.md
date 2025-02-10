<p align="center" style="font-size:40px;">
<b>PCLA: A framework for testing autonomous agents in CARLA simulator</b>
</p>

<p align="center">
PCLA (Pretrained CARLA Leaderboard Agent) is a versatile framework that allows you to utilize the autonomous agents from the <a href="https://leaderboard.carla.org/leaderboard/">CARLA Leaderboard</a> independently of its core codebase and put them on your vehicle. </br>

* PCLA provides a clear method to deploy ADAs onto a vehicle without relying on the Leaderboard codebase.
* Enables easy switching between ADAs without requiring changes to CARLA versions or programming environments.
* Allows you to have multiple vehicles with different autonomous agents.
* Provides the next movement action computed by the chosen agent, which can then be utilized in any desired application.
* Is fully compatible with the latest version of CARLA and independent of the Leaderboard’s specific CARLA version.
* Includes nine different high-performing ADAs trained with 17 distinct training seeds. 
</br>
Paper available at <a href="#">change this</a>
</p>

<p align="center">
**PCLA was tested on Linux Ubuntu 22 and CARLA 9.15 Unreal Engine 4**
</p>

<p align="center">
A video tutorial on how to use PCLA is available below. </br>
<div align="center">
  <a href="https://www.youtube.com/watch?v=QyaMK6vclBg"><img src="https://img.youtube.com/vi/QyaMK6vclBg/0.jpg" alt="PCLA Video Tutorial"></a>
</div>

</p>


## Contents

1. [Setup](#setup)
2. [Pre-Trained Weights](#pre-trained-weights)
3. [Autonomous Agents](#autonomous-agents)
4. [How to Use](#how-to-use)
5. [Environment Variables](#environment-variables)
6. [Sample Code](#sample-code)
7. [FAQ](#FAQ)
8. [Citation](#citation)

## Setup
Download and install the <a href="https://carla.readthedocs.io/en/latest/">CARLA simulator</a> from the official website. Based on your preference, you can either use quick installation or build from source.</br>
Clone the repository and build the conda environment:
```Shell
git clone https://github.com/MasoudJTehrani/PCLA
cd PCLA
conda env create -f environment.yml
conda activate PCLA
```
Alternatively, you can use the `requirements.txt` file to install the required libraries by using:</br>
```Shell
pip install -r requirements.txt
```
Please make sure CUDA and PyTorch are installed.</br>
<a href="https://www.gpu-mart.com/blog/install-nvidia-cuda-11-on-ubuntu">Tutorial for installing CUDA on ubuntu<a></br>
<a href="https://pytorch.org/get-started/locally/">Tutorial for PyTorch<a>


## Pre-Trained Weights

Download the pre-trained weights from <a href="https://zenodo.org/records/14446470">Zenodo</a> or directly from <a href="https://zenodo.org/records/14446470/files/pretrained.zip?download=1">here</a> and extract them into the `PCLA/agents/` directory.</br> 
Ensure that each folder of pre-trained weights is placed directly next to its respective model's folder. The `agents` folder should look like this.
```Bash
├── agents
   ├── garage
   ├── garagepretrained
   ├── interfuser
   ├── interfuserpretrained
   ├── neat
   └── neatpretrained
```

## Autonomous Agents

PCLA includes 9 different autonomous agents and 17 distinct training seeds to choose from.
- **CARLA-garage**
  - Contains 4 different autonomous agents with 3 training seeds for each agent. To use these agents you need to set some [Environment Variables](#environment-variables).
    - *garage_lav_#*, replace # with the seed number from 0 to 2.
    - *garage_aim_#*, replace # with the seed number from 0 to 2.
    - *garage_ld_#*, replace # with the seed number from 0 to 2. This is to use their leaderboard agent.
    - *garage_l6_#*, replace # with the seed number from 0 to 2. This is to use their longest6 agent.
  - Repository: [https://github.com/autonomousvision/carla_garage](https://github.com/autonomousvision/carla_garage)
- **Neat**
  - Contains 4 different autonomous agents.
      - *neat_neat*
      - *neat_aimbev*
      - *neat_aim2dsem*
      - *neat_aim2ddepth*
  - Repository: [https://github.com/autonomousvision/neat](https://github.com/autonomousvision/neat)
- **Interfuser**
  - Contains 1 autonomous agent. To use this agent you need to set an [Environment Variables](#environment-variables).
     -*if_if*
  - Repository: [https://github.com/opendilab/InterFuser](https://github.com/opendilab/InterFuser)

## How to Use
First, run CARLA. You don't need any special arguments.
```Shell
./CarlaUE4.sh
```
Then open another terminal and run your code.</br>
To use PCLA, simply import it and use the PCLA class to define an autonomous vehicle with your chosen autonomous agent.
```Shell
from PCLA import PCLA

agent = "neat_neat"
route = "./sampleRoute.xml"
pcla = PCLA(agent, vehicle, route, world, client)

ego_action = pcla.get_action()
vehicle.apply_control(ego_action)
```
In the code above, the agent is your chosen autonomous agent. You can choose your agent from the list of [Autonomous Agents](#autonomous-agents).</br>
You also need to pass the `route` that you want your vehicle to follow. The route should be in the format of the Leaderboard waypoints as an `XML` file.</br>
To make it easy, PCLA provides you with a function called `routeMaker` that gets an array of <a href="https://carla.readthedocs.io/en/latest/core_map/#waypoints">CARLA waypoints</a>, reformats it to a Leaderboard format and save it as an XML file.</br>
Example of generating XML route from a list of <a href="https://carla.readthedocs.io/en/latest/core_map/#waypoints">CARLA waypoints</a>:
```Shell
from PCLA import routeMaker

client = carla.Client('localhost', 2000)
world = client.get_world()
mp = world.get_map()
waypoints = mp.generate_waypoints(2)
routeMaker(waypoints, "route.xml")
```
The other arguments you have to pass to PCLA are the world, the client, and the vehicle you want to put the agent on. </br>
To get one action in a frame from the agent and apply it to your vehicle you can call the `pcla.get_action` method. </br>
Example:
```Shell
ego_action = pcla.get_action()
vehicle.apply_control(ego_action)
```
### Environment Variables
Carla_garage and Interfuser require you to set an environment variable before using their agents.
Environment variables for each agent are:
- **garage_lav_#**
  ```Shell
  export STOP_CONTROL=1
  ```
- **garage_aim_#**
  ```Shell
  export DIRECT=0
  ```
- **garage_ld_#**
  ```Shell
  export DIRECT=0
  ```
- **garage_l6_#**
  ```Shell
  export UNCERTAINTY_THRESHOLD=033
  ```
- **if_if**
  ```Shell
  export ROUTES=path_to_route.xml
  ```
  The path to the same XML route file you used in PCLA

Remember to unset a variable before using another agent.
```Shell
unset DIRECT
```

## Sample Code
A sample code is provided for you to test PCLA. Just go to the PCLA directory and run:
```Shell
python sample.py
```
This sample is in Town02 of the CARLA simulator.

## FAQ
Frequently asked questions and possible issues are solved in <a href="https://github.com/MasoudJTehrani/PCLA/issues?q=is%3Aissue+is%3Aclosed">the issues section</a>.

## Citation
If you find PCLA useful, please consider giving it a star &#127775;,
and cite the published paper:
```Shell
bibtex citation
```
