# SC2 Predictor

## Workflow for interestingness regression
1. get-videos to download from youtube and get images
2. classify-thumbnails to classify (manually or with existing neural network)
3. prepare-files-for-learning to copy images into folders
4. learn-regression to build model for interestingness prediction

## Workflow for ingame classification
1. get-vdieos to download from youtube and get images
2. classify-thumbnails to classify images
3. train-ingame-classifier to build model for ingame classification

## ingame classification:
- 0: in-game
- 1: everything out of game

## interestingness regression:
- 0: nothing of any interest
- 1: opposing units on screen
- 2: small fight
- 3: big fight
- 4: craziest fight