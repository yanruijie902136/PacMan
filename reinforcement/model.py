import nn

class DeepQNetwork():
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.1
        self.numTrainingGames = 6000
        self.batch_size = 256

        layer1_size, layer2_size = 100, 50
        self.W1, self.b1 = nn.Parameter(self.state_size, layer1_size), nn.Parameter(1, layer1_size)
        self.W2, self.b2 = nn.Parameter(layer1_size, layer2_size), nn.Parameter(1, layer2_size)
        self.W3, self.b3 = nn.Parameter(layer2_size, self.num_actions), nn.Parameter(1, self.num_actions)
        self.parameters = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def set_weights(self, layers):
        self.parameters = []
        for i in range(len(layers)):
            self.parameters.append(layers[i])

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted
        by the network, and Q_target.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(states), Q_target)

    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            result: (batch_size x num_actions) numpy array of Q-value
                scores, for each of the actions
        """
        "*** YOUR CODE HERE ***"
        x = nn.ReLU(nn.AddBias(nn.Linear(states, self.W1), self.b1))
        x = nn.ReLU(nn.AddBias(nn.Linear(x, self.W2), self.b2))
        return nn.AddBias(nn.Linear(x, self.W3), self.b3)

    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        "*** YOUR CODE HERE ***"
        loss = self.get_loss(states, Q_target)
        gradients = nn.gradients(loss, self.parameters)
        for parameter, gradient in zip(self.parameters, gradients):
            parameter.update(gradient, -self.learning_rate)
