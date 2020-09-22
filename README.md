# Q-Learning2DMaze
My code implementation with comments, class structure and terminal arguments for finding optimal route in 2d Maze.
![Alt Text](https://github.com/rafmph/Q-Learning2DMaze/blob/master/mazeexample.gif)

[Gym maze environment by MattChanTK must be installed](https://github.com/MattChanTK/gym-maze)

Code executes from terminal by typing `python3 reinforcement_maze.py --episodes 50000 --iterations 200 --explore_r 0.001 --discount_r 10` these arguments can also be ommited and default hyperparams will be chosen.

After entering command in terminal you'll be prompted to choose 12 different available environments:

1. 2x2 Same structure maze without portals
2. 3x3 Same structure maze without portals
3. 3x3 Random structure maze without portals
4. 5x5 Same structure maze without portals
5. 5x5 Random structure maze without portals
6. 10x10 Same structure maze without portals
7. 10x10 Random structure maze without portals
8. 10x10 Random structure maze WITH portals
9. 20x20 Random structure maze WITH portals
10. 30x30 Random structure maze WITH portals
11. 100x100 Same structure maze without portals
12. 100x100 Random structure maze without portals
