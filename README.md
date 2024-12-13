<p align="center" style="font-size:40px;">
<b>PCLA: A framework for testing autonomous agents in CARLA simulator</b>
</p>

<p align="center" style="font-size:20px;">
PCLA (Pretrained CARLA Leaderboard Agent) is a versatile framework that allows you to utilize the autonomous agents from the <a href="https://leaderboard.carla.org/leaderboard/">CARLA Leaderboard</a> independently of its core codebase and put them on your vehicle. With PCLA, you can have with your own CARLA API code and then easily deploy your selected autonomous agents to your vehicles. Additionally, PCLA lets you work with the latest version of CARLA, freeing you from the limitations of using a specific version.
</p>

This repository
was tested on Linux ubuntu 22 and carla 9.15
ValueError: An output representation was chosen that was not trained. : unset DIRECT

## Contents

1. [Setup](#setup)
2. [Pre-Trained Weights](#pre-trained-weights)
3. [Autonomous Agents](#autonomous-agents)
4. [How to Use](#how-to-use)
5. [Environment Variables](#environment-variables)
6. [Sample Code](#sample-code)
7. [Common Mistakes](#common-mistakes)
8. [Citation](#citation)
9. [Acknoledgements](#acknoledgements)

## Setup
Download and install the <a href="https://carla.readthedocs.io/en/latest/">CARLA simulator</a> from the official website. Based on your preference, you can either use quick installation or build from source.</br>
Clone the repository and build the conda environment:
```Shell
git clone https://github.com/MasoudJTehrani/PCLA
cd PCLA
conda env create -f environment.yml
conda activate PCLA
```

## Pre-Trained Weights

Download the pre-trained weights from ????[insert link here] and extract them into the `PCLA/agents/` directory.</br> 
Ensure that each folder of pre-trained weights, is placed directly next to its respective model's folder.

## Autonomous Agents

PCLA includes 9 different autonomous agents and 17 distinct training seeds to choose from.
- **CARLA-garage**
  - Contains 4 different autonomous agents with 3 training seeds for each agent. To use these agents to need to set some [Environment Variables](#environment-variables).
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
  - Contains 1 autonomous agent. To use this agent to need to set an [Environment Variables](#environment-variables).
     -*if_if*
  - Repository: [https://github.com/opendilab/InterFuser](https://github.com/opendilab/InterFuser)

## How to Use

To use PCLA, simply import it and use the PCLA class to define an autonomous vehicle with your chosen autonomous agent.
```Shell
from PCLA import PCLA

agent = "garage_l6_2"
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
The other arguments you have to pass to PCLA is the world, client and the vehicle you want to put the agent on. </br>
To get one action in a frame from the agent and apply it to your vehicle you can call the `pcla.get_action` method. </br>
Example:
```Shell
ego_action = pcla.get_action()
vehicle.apply_control(ego_action)
```
