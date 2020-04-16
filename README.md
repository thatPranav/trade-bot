# trading-bot

Task : To develop an AI to predict the stock prices and accordingly decide on buying, selling or holding stock. 

* Key points to consider:
    * It is a continuous task
	* It involves dynamic programming 
	* The agent impacts on the state of environment


* Approach Used: 
    * Deep Q learning :	
        * Initialize Q Table
        * Choose an action
        * Perform Action
        * Measure Reward
        * Update Q table
	
	
	* Using Bellman Equation 
	* Implementation is done using pytorch.
	* The Agent can take three actions :
        * Buy 
        * Hold 
        * Sell

    * The dataset used is an opensource dataset
	* The model has a three layer neural network.
