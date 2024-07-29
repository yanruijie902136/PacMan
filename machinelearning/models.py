import backend
import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        return 1 if nn.as_scalar(self.run(x)) >= 0 else -1

    def train(self, dataset: backend.PerceptronDataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        while True:
            hasMistakes = False
            for x, y in dataset.iterate_once(batch_size=1):
                y_scalar = nn.as_scalar(y)
                if self.get_prediction(x) != y_scalar:
                    hasMistakes = True
                    self.w.update(direction=x, multiplier=y_scalar)
            if not hasMistakes:
                return

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # Hyperparameters.
        self.batch_size = 8
        self.learning_rate = 0.01

        # Trainable parameters.
        layer_size1, layer_size2 = 100, 100
        self.W1, self.b1 = nn.Parameter(1, layer_size1), nn.Parameter(1, layer_size1)
        self.W2, self.b2 = nn.Parameter(layer_size1, layer_size2), nn.Parameter(1, layer_size2)
        self.W3, self.b3 = nn.Parameter(layer_size2, 1), nn.Parameter(1, 1)

        self.parameters = [
            self.W1, self.b1,
            self.W2, self.b2,
            self.W3, self.b3,
        ]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        x = nn.ReLU(nn.AddBias(nn.Linear(x, self.W1), self.b1))
        x = nn.ReLU(nn.AddBias(nn.Linear(x, self.W2), self.b2))
        return nn.AddBias(nn.Linear(x, self.W3), self.b3)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset: backend.RegressionDataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            total_train_loss, num_batches = 0, 0
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                total_train_loss += nn.as_scalar(loss)
                num_batches += 1

                gradients = nn.gradients(loss, self.parameters)
                for parameter, gradient in zip(self.parameters, gradients):
                    parameter.update(gradient, -self.learning_rate)

            average_train_loss = total_train_loss / num_batches
            # print("average train loss:", average_train_loss)
            if average_train_loss < 0.02:
                return

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # Hyperparameters.
        self.batch_size = 32
        self.learning_rate = 0.1

        # Trainable parameters.
        input_size, output_size = 784, 10
        layer1_size, layer2_size = 200, 100
        self.W1, self.b1 = nn.Parameter(input_size, layer1_size), nn.Parameter(1, layer1_size)
        self.W2, self.b2 = nn.Parameter(layer1_size, layer2_size), nn.Parameter(1, layer2_size)
        self.W3, self.b3 = nn.Parameter(layer2_size, output_size), nn.Parameter(1, output_size)

        self.parameters = [
            self.W1, self.b1,
            self.W2, self.b2,
            self.W3, self.b3,
        ]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        x = nn.ReLU(nn.AddBias(nn.Linear(x, self.W1), self.b1))
        x = nn.ReLU(nn.AddBias(nn.Linear(x, self.W2), self.b2))
        return nn.AddBias(nn.Linear(x, self.W3), self.b3)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset: backend.DigitClassificationDataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, self.parameters)
                for parameter, gradient in zip(self.parameters, gradients):
                    parameter.update(gradient, -self.learning_rate)

            if self.learning_rate > 0.005:
                self.learning_rate -= 0.005

            if dataset.get_validation_accuracy() >= 0.98:
                return

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 32
        self.learning_rate = 0.1
        self.hidden_size = 100

        self.W = nn.Parameter(self.num_chars, self.hidden_size)
        self.W_hidden = nn.Parameter(self.hidden_size, self.hidden_size)
        self.W_output = nn.Parameter(self.hidden_size, len(self.languages))

        self.parameters = [self.W, self.W_hidden, self.W_output]

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        for i, x in enumerate(xs):
            z = nn.Linear(x, self.W)
            if i > 0:
                z = nn.Add(z, nn.Linear(h, self.W_hidden))
            h = nn.ReLU(z)
        return nn.Linear(h, self.W_output)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset: backend.LanguageIDDataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, self.parameters)
                for parameter, gradient in zip(self.parameters, gradients):
                    parameter.update(gradient, -self.learning_rate)

            if dataset.get_validation_accuracy() >= 0.88:
                return
