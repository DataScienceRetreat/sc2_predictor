# SC2 Predictor

## What is this?
### WHY
A huge amount of video content is created and uploaded to the internet every day.
Selecting highlights increases user engagement and decreases information overload, but at scale it cannot be done by hand.

Video games are complex virtual environments, limited by parameters such as camera angles and available units. They are a well-known testbed for machine learning models and used heavily in reinforcement learning.

### WHAT
This projects predicts the interestingness of a moment in Starcraft 2, a popular real-time strategy game with a dedicated community in the growing eSports market. Two players fight against each other in 5 minutes to hour-long games. Players compete with one of three different races in terms of strategy and timely execution, often averaging 300 input actions with keyboard and mouse per minute.

A highlight or interesting moment has multiple interpretations. 
This project is concerned with the visual fight sequences between the armies of two players.

The goal was a pipeline which reads a video link and outputs the interestingness over time. 
A peak in interestingness over a short time signals an exciting moment, short highlight clips are selected based on these change in the time series. 
The pipeline includes two neural networks, one for recognising in-game images and one for the regression output.

Read more on [Medium](https://medium.com/highlighthero/ai-powered-highlights-9f4d55986445#.5dhc4ade3)
[![Finding highlights in videogame video replays - Data Science Retreat portfolio project](http://img.youtube.com/vi/cUtNtR18sEY/0.jpg)](https://www.youtube.com/watch?v=cUtNtR18sEY "Finding highlights in videogame video replays - Data Science Retreat portfolio project")

## How can I use this?

### Workflow for interestingness regression
1. Download videos using ```get-videos.py``` with text file with youtube links of replays (for example: ```video_list.txt```)
2. Run ```classify-thumbnails.py`` to classify images (manually or with existing neural network)
3. Run ```prepare-files-for-learning.py``` to copy images into their correct folders
4. Run ```learn-regression.py``` to train a model for interestingness prediction

### Workflow for ingame classification
1. Download videos using ```get-videos.py``` with text file with youtube links of replays (for example: ```video_list.txt```)
2. Run ```classify-thumbnails.py`` to classify images (manually or with existing neural network)
3. Run ```learn-regression.py``` to train a model for ingame classification

## ingame classification:
- 0: in-game
- 1: everything out of game

## interestingness regression:
- 0: nothing of any interest
- 1: opposing units on screen
- 2: small fight
- 3: big fight
- 4: craziest fight