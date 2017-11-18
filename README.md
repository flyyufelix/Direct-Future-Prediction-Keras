# Discounted Future Prediction (DFP) implemented in Keras

This repo includes implementation of Discounted Future Prediction (DFP) Algorithm describe in [this paper](https://arxiv.org/pdf/1611.01779.pdf). The implementation is tested on the [VizDoom](http://vizdoom.cs.put.edu.pl/) **Health Gathering** scenario, which is a 3D partially observable environment.

For a general introduction of DFP and step-by-step walkthrough of the implementation, please check out my blog post at https://flyyufelix.github.io/2017/11/17/direct-future-prediction.html.

<img src="/resources/medkit_pickup.gif" width="300">

## Results

Below is the performance chart of 40,000 episodes of **DFP** and **DDQN** running on Health Gathering. Y-axis is the average survival time (moving average over 50 episodes).

![DFP Performance Chart](/resources/dfp_chart.png)

## Usage

First follow [this](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md) instruction to install VizDoom. If you use python, you can simply do pip install:

```
$ pip install vizdoom
```

Second, clone [ViZDoom](https://github.com/mwydmuch/ViZDoom) to your machine, copy the python files provided in this repo over to `examples/python`.

Next, edit `scenarios/health_gathering.cfg` file. Replace this line 
```
doom_scenario_path = health_gathering.wad
```
with
```
doom_scenario_path = health_gathering_supreme.wad
```

To test if the environment is working, run

```
$ cd examples/python
$ python dfp.py
```

You should see some printouts indicating that the DFP is running successfully. Errors will be thrown otherwise.

## Dependencies

* Keras 1.2.2 / 2.0.5
* Tensorflow 0.12.0 / 1.2.1
* [VizDoom Environment](http://vizdoom.cs.put.edu.pl/)
