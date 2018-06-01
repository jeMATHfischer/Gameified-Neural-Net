# Gameified-Neural-Net

We consider a game theory motivated approach to neural nets. Suppose we dispose of a feed forward neural network with layer $\mathcal{L}_1,\hdots,\mathcal{L}_N$ and a datset $\mathcal{D}$ of $m$ measurements which we denote by $\mathcal{D} = (\mathcal{D}_1^0|\hdots|\mathcal{D}_{\lambda}^0)$. The power 0 denotes that it is the data which we are going to feed in the first layer of the NN.
The problem we want to tackle is a supervisied classification task, i.e., every measurement desposes of a corresponding label. Furthermore, let $\mathcal{H}$ be the loss function for our problem. Lastly we denote the weight matrix between layer $\mathcal{L}_i$ and $\mathcal{L}_{i+1}$ in the sth step as $W^{i,s} and activation function $\sigma^i$.
We consider the Layers as players in a sequential game and their weight matrices as their strategies. 
The approach is motivated by a sequential game where the first player 1 gives to the second player 2 information in such a way that given that 2 and all subsequent players do not change their strategies, the outcome will be optimal. 1 can achieve this by changing his weights accordingly.
  

