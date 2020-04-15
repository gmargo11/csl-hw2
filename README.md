# CSL HW2

This repository contains code for MIT 6.884 Homework #2. My writeup is in [writeup.pdf](https://github.com/gmargo11/csl-hw2/blob/master/writeup.pdf).

All code will run out of the box in the airobot docker (available at https://github.com/Improbable-AI/airobot)

### Inverse Model Learning
1. run `python inverse_model.py` to learn and evaluate an inverse model for pushing the object.
2. run `bash make_inverse_video.sh` to produce webm of results

### Forward Model Learning
1. run `python forward_model.py` to learn and evaluate the forward model for pushing the object.
2. run `bash make_forward_video.sh` to produce webm of results

### Two-Push Experiment
1. run `python two_pushes.py` to perform the two-push experiment on both models.
2. run `bash make_two_push_video.sh` to produce webm of results
