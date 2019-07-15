# Battleship
My task was to build an AI to play battleship using machine learning. My major focus was on building a framework and just using a simple model that works well enough. With more time I could improve the model and expand the scope of the AI to also include setting up the board, not just playing. My solution uses a RandomForestClassifier with 1000 trees and max depth of 10. These hyper-parameters were selected based on prior experience. I trained the model overnight on 100,000 randomly generated game boards.

I've modeled the board with values per position of water=0, hidden=1, or boat=2. There is a 'hidden' board with the actual boat placement info and there's an 'observed' board that's what the AI sees. The features the model sees when classifying include the 'observed' values of each position up to two manhattan distance away from the tile, as well as it's x and y coordinates, and how many hits and misses there are so far in the game. With each turn, the model predicts for all 100 positions and the first available position with the best score gets revealed next. If the AI guesses right it'll reveal a boat, otherwise it'll reveal water.

This AI does not set up the board / place boats, right now that's just a simple random algorithm. I've noticed the boat placing algorithm does not place boats like a human since humans as far as I've seen usually avoid putting boats against each other, but the random algorithm does not mind!

With a fresh game, the model seems to like picking positions along the diagonals until it makes a hit. Once it's got a hit it seems to look for boats adjacent to the tile until it hits water on both ends. Unlike a human it also seems to really like looking for boats placed adjacent to each other. Not exactly sure why it does that, but I'm guessing it does that because the random boat placement algorithm is a bit clumpy.


# Future directions:
* Create an AI to place the boats. Train it to best fool the AI I already implemented.
* Take the boat placing one level further and implement an adversarial loop that has one model play the other. This is reminiscent of AlphaGo, AlphaStar, etc.
