This is a work in progress. Any ideas welcomed !

# Installation

## To install all dependancies
```
pip install -r requirements.txt
```
## To install Sokoban environment
```
git clone https://github.com/mpSchrader/gym-sokoban.git
cd gym-sokoban
pip install -e .
```

# Run and test

See **main.ipynb** for details.

# Description

This implementation tries to follow "Learning to search with MCTSnets" (Guez et al.) [1] as closely as possible.
All mentioned networks are available in **MCTSnet/models/**.
We focused on a simple random policy as a first step and approximate computation for the policy is not implemented. This should however provide good results according to the authoers.

**MCTSnet/models/MCTSnet.py** implements all the search logic.

The code tends to be quite general and should be easily adapted for any environment (we tried to use a inteerface as close as gym as possible).

**MCTSnet/trainer.py** provides an easy way to train, test and play games.
Two games were tested:
 - the Sokoban, as per the original article
 - the MouseGame, a simple game were a mouse need to eat the cheese and avoid poison

The algorithm has proven to converge on the mousegame, however event after 15hours of training the MCTSnet performance is poor. Additionnal testing is required.


# References
[1] Arthur Guez, Theophane Weber, Ioannis Antonoglou, Karen Simonyan, Oriol Vinyals, Daan Wierstra, Remi Munos, and David Silver. Learning
to  search  with  mctsnets. CoRR,  abs/1802.04697, 2018.